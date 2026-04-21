import json
import numpy as np
import sys
import wandb
import time
from ml_collections import config_flags
from typing import Any, Literal, Dict
import tensorflow as tf
from PIL import Image
sys.path.append(".")
import numpy as np
from flask import Flask, request, jsonify
from flask_ngrok import run_with_ngrok
import pyngrok 
import ngrok
import base64
from io import BytesIO
from PIL import Image
from absl import flags
from fastapi import FastAPI
import uvicorn

# Palivla
from palivla.model_components import ModelComponents
from palivla.inference import run_inference, make_sharding

# Jax imports
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp

print("Inference server running...")
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(physical_devices, "GPU")
print("VISIBLE DEVICES: ", jax.devices())

tf.random.set_seed(jax.process_index())
wandb.login()
run = wandb.init(
    # Set the project where this run will be logged
    project="cast-inference",
    mode="online",
)

# CLI FLAGS
config_flags.DEFINE_config_file(
        "config", "/home/lovelace/CAST/cast-vla/configs/bridge_cast_config.py", "Path to the config file."
)
flags.DEFINE_string("platform", "gpu", "Platform to run on.")
flags.DEFINE_string("checkpoint_dir", "gs://cat-logs/bridge_cast_2026_04_14_21_09_03", "Path to the checkpoint directory.")
flags.DEFINE_integer("checkpoint_step", 135000, "Step to resume from.")
flags.DEFINE_string("prompt", "", "Prompt to generate action from.")
flags.DEFINE_string("host", "0.0.0.0", "Host to run on.")
flags.DEFINE_integer("port", 5000, "Port to run on.")
flags.DEFINE_string(
    "dataset_statistics",
    None,
    "Path to dataset_statistics JSON (local or gs://). Must match training normalization. "
    "Supports (1) flat octo cache style {\"action\": {...}, ...} or (2) per-dataset dicts from "
    "train.py --stats_only, e.g. {\"bridge_dataset\": {\"action\": ...}, ...}. "
    "Use --statistics_dataset when multiple dataset keys exist.",
)
flags.DEFINE_string(
    "statistics_dataset",
    None,
    "When the JSON has one top-level key per dataset (e.g. bridge_dataset), select which "
    "dataset's action statistics to use for unnormalization. If omitted and only one such "
    "key exists, that key is used; if multiple exist, bridge_dataset is preferred when present.",
)
flags.DEFINE_enum(
    "action_normalization",
    "bounds",
    ["bounds", "normal"],
    "Inverse transform for actions. Training uses BOUNDS in octo.data.dataset.make_dataset_from_rlds "
    "(see octo/octo/data/dataset.py: the parameter from config is currently overridden). "
    "Use 'normal' only if your pipeline used mean/std normalization with no override.",
)


# Bridge / bridge_cast OXE configs use EEF_POS: 6 DoF + gripper; gripper is not normalized when mask is set.
_DEFAULT_EEF_POS_ACTION_MASK = np.array([True] * 6 + [False], dtype=bool)


def load_dataset_statistics(path: str) -> dict[str, Any]:
    """Load dataset statistics JSON (same schema as octo ``get_dataset_statistics``)."""
    with tf.io.gfile.GFile(path, "r") as f:
        return json.load(f)


def resolve_action_statistics_for_unnormalize(
    raw: dict[str, Any],
    statistics_dataset: str | None = None,
) -> tuple[dict[str, Any], str]:
    """Return a stats dict suitable for ``unnormalize_actions`` (must have top-level ``action``).

    - **Flat** (legacy cache): ``{"action": {"mean": ...}, "num_transitions": ...}``
    - **Per-dataset** (``train.py --stats_only``): ``{"bridge_dataset": {"action": ...}, ...}``
    """
    top_action = raw.get("action")
    if (
        isinstance(top_action, dict)
        and "mean" in top_action
    ):
        return raw, "flat"

    candidate_keys = [
        k
        for k, v in raw.items()
        if isinstance(v, dict)
        and isinstance(v.get("action"), dict)
        and "mean" in v["action"]
    ]
    if not candidate_keys:
        raise ValueError(
            "dataset_statistics JSON must either include top-level "
            "'action' with 'mean', or per-dataset objects each with "
            f"'action.mean'. Top-level keys: {list(raw.keys())!r}"
        )
    if statistics_dataset is not None:
        if statistics_dataset not in raw:
            raise ValueError(
                f"--statistics_dataset={statistics_dataset!r} not in JSON. "
                f"Available: {candidate_keys!r}"
            )
        inner = raw[statistics_dataset]
        if not isinstance(inner.get("action"), dict):
            raise ValueError(f"No action block under {statistics_dataset!r}")
        return inner, statistics_dataset
    if len(candidate_keys) == 1:
        k = candidate_keys[0]
        return raw[k], k
    if "bridge_dataset" in candidate_keys:
        return raw["bridge_dataset"], "bridge_dataset"
    raise ValueError(
        f"Multiple per-dataset statistics entries {candidate_keys}; "
        "set --statistics_dataset to one of these names."
    )


def unnormalize_actions(
    actions: np.ndarray,
    dataset_statistics: dict[str, Any],
    *,
    normalization_type: Literal["bounds", "normal"] = "bounds",
    eps: float = 1e-8,
) -> np.ndarray:
    """Invert ``normalize_action_and_proprio`` from ``octo.data.utils.data_utils``.

    Args:
        actions: Normalized action vector, shape (..., A).
        dataset_statistics: Must include ``action`` with ``mean``, ``std``, and for ``bounds`` also ``p01``, ``p99``.
        normalization_type: ``bounds`` maps [-1, 1] (clipped during training) back to raw space; ``normal`` inverts z-score.
        eps: Same epsilon as training (1e-8) for division.

    Returns:
        Actions in the same raw space as before training normalization.
    """
    x = np.asarray(actions, dtype=np.float64)
    a = dataset_statistics["action"]
    mean = np.asarray(a["mean"], dtype=np.float64)
    std = np.asarray(a["std"], dtype=np.float64)
    if "mask" in a:
        mask = np.asarray(a["mask"], dtype=bool)
    else:
        mask = np.broadcast_to(_DEFAULT_EEF_POS_ACTION_MASK, mean.shape)

    if normalization_type == "normal":
        # Forward: (raw - mean) / (std + eps); inverse: raw = x * (std + eps) + mean
        return np.where(mask, x * (std + eps) + mean, x)

    if normalization_type == "bounds":
        # Forward: clip(2 * (raw - p01) / (p99 - p01 + eps) - 1, -1, 1); inverse (linear region):
        p01 = np.asarray(a["p01"], dtype=np.float64)
        p99 = np.asarray(a["p99"], dtype=np.float64)
        denom = p99 - p01 + eps
        raw = p01 + (x + 1.0) * 0.5 * denom
        return np.where(mask, raw, x)

    raise ValueError(f"Unknown normalization_type: {normalization_type!r}")


class PolicyServer: 
    def __init__(
        self,
        policy_dir: str,
        policy_ckpt: str,
        platform: str,
        config: str,
        input_prompt: str,
    ):
        self.platform = platform
        self.policy_dir = policy_dir
        self.policy_ckpt = policy_ckpt
        self.config = config
        self.input_prompt = input_prompt
        self.action_stats: dict[str, Any] | None = None
        self._action_norm: Literal["bounds", "normal"] = "bounds"

        if self.platform == "tpu":
            jax.distributed.initialize()
        sharding_metadata = make_sharding(self.config)

        print("\nLoading model...", self.policy_dir)
        self.model = ModelComponents.load_static(self.policy_dir, sharding_metadata, weights_only=True)
        self.manager = ocp.CheckpointManager(self.policy_dir, options=ocp.CheckpointManagerOptions())
        self.model.load_state(self.policy_ckpt, self.manager, weights_only=True)
        print("\nModel loaded!")

    def set_action_dataset_statistics(
        self,
        path: str | None,
        normalization: Literal["bounds", "normal"],
        statistics_dataset: str | None = None,
    ) -> None:
        """Load stats for ``unnormalize_actions`` (optional)."""
        if not path:
            print(
                "No --dataset_statistics: returning policy outputs in normalized space "
                f"(expected format: {normalization})."
            )
            self.action_stats = None
            self._action_norm: Literal["bounds", "normal"] = normalization
            return
        raw = load_dataset_statistics(path)
        resolved, which = resolve_action_statistics_for_unnormalize(
            raw, statistics_dataset=statistics_dataset
        )
        self.action_stats = resolved
        self._action_norm = normalization
        print(
            f"Loaded dataset statistics for action unnormalization from {path} "
            f"({normalization}, entry={which!r})."
        )

    def sample_action(self, payload: Dict[str, Any]):
        print("Getting action...")
        image = Image.open(BytesIO(base64.b64decode(payload["image"])))
        image.save("image.png")
        image = np.array(image)
        proprio = np.array(payload["proprio"])
        prompt = np.array([payload["instruction"].encode("utf-8")])
        
        if prompt == "": 
            prompt = self.input_prompt
        
        print(f"Prompt: {prompt}")
        
        # Make sure we use a batch size of 1
        print(f"Image shape: {image.shape}")
        print(f"Proprio shape: {proprio.shape}")
        print(f"Prompt shape: {prompt.shape}")
        if len(image.shape) < 5:
            while len(image.shape) < 5:
                image = np.expand_dims(image, 0)
            assert image.shape == (1, 1, 224, 224, 3)
        
        if len(proprio.shape) < 3:
            while len(proprio.shape) < 3:
                proprio = np.expand_dims(proprio, 0)
            assert proprio.shape == (1, 1, 7)
        
        assert prompt.shape == (1,)
        
        if self.platform == "tpu":
            image = image.repeat(4, axis=0)
            proprio = proprio.repeat(4, axis=0)
            prompt = prompt.repeat(4, axis=0)
            action = np.random.randn(4, 1, 1, 7).astype(np.float64)
        
        else:
            action = np.random.randn(1, 1, 1, 7).astype(np.float64)

        batch = {"task" : 
                        {
                            "language_instruction" : prompt, 
                            "pad_mask_dict": 
                                {
                                    "language_instruction": np.ones(prompt.shape, dtype=bool)
                                },
                            },
                 "observation": 
                        {
                            "image_primary": image, 
                            "proprio": proprio,
                            "pad_mask_dict": 
                                {
                                    "image_primary": np.ones(image.shape, dtype=bool), 
                                    "proprio": np.ones(proprio.shape, dtype=bool)
                                }
                        },
                 "action": action,    
                }

        # Run inference
        start_time = time.time()
        action_horizon = 1

        # Predict the output 
        if self.config.get("sampler") is not None:
            sampler = self.config["sampler"]
            if sampler == "greedy":
                temperature = None
            else:
                temperature = self.config["temperature"]
        else:
            sampler = "greedy"
            temperature = None
            
        predicted_actions, actions_mask, tokens = self.model.predict(batch, action_dim=7, action_horizon=action_horizon, return_tokens=True, include_action_tokens=False, sampler=sampler, temperature=temperature)
        
        total_time = time.time() - start_time
        print(f"Total time: {total_time}s")
        
        if len(predicted_actions.shape) > 1: 
            while len(predicted_actions.shape) > 1:
                predicted_actions = predicted_actions.squeeze(0)
        assert predicted_actions.shape == (7,)
        out = np.asarray(predicted_actions)
        if self.action_stats is not None:
            out = unnormalize_actions(
                out, self.action_stats, normalization_type=self._action_norm
            )
        print(f"Action: {out.tolist()}")
        return {"action": out.tolist()}

    def run(self, host: str = "0.0.0.0", port: int = 5000) -> None:
        self.app = FastAPI()
        self.app.post("/act")(self.sample_action)
        uvicorn.run(self.app, host=host, port=port)
        

if __name__ == "__main__":
    
    FLAGS = flags.FLAGS
    FLAGS(sys.argv) 

    # Start the server
    server = PolicyServer(
        policy_dir=flags.FLAGS.checkpoint_dir,
        policy_ckpt=flags.FLAGS.checkpoint_step,
        platform=flags.FLAGS.platform,
        config=flags.FLAGS.config,
        input_prompt=flags.FLAGS.prompt,
    )
    server.set_action_dataset_statistics(
        flags.FLAGS.dataset_statistics,
        flags.FLAGS.action_normalization,
        statistics_dataset=flags.FLAGS.statistics_dataset,
    )
    server.run(host=flags.FLAGS.host, port=flags.FLAGS.port)