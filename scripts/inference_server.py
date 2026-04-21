import base64
import time
from io import BytesIO
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import tensorflow as tf
import uvicorn
import wandb
from absl import flags
from fastapi import FastAPI
from ml_collections import config_flags
from PIL import Image

import sys

sys.path.append(".")

from palivla.inference import make_sharding, run_inference
from palivla.model_components import ModelComponents

print("Inference server running...")
physical_devices = tf.config.list_physical_devices("GPU")
tf.config.set_visible_devices(physical_devices, "GPU")
print("VISIBLE DEVICES: ", jax.devices())

tf.random.set_seed(jax.process_index())
wandb.login()
run = wandb.init(
    project="cast-inference",
    mode="online",
)

config = None
model = None
avg_time: list[float] = []
input_prompt = ""


def _ensure_model_loaded():
    global config, model, input_prompt
    if model is not None:
        return
    f = flags.FLAGS
    config = f.config
    input_prompt = f.prompt

    if f.platform == "tpu":
        jax.distributed.initialize()
    sharding_metadata = make_sharding(config)

    print("\nLoading model...", f.checkpoint_dir)
    model = ModelComponents.load_static(
        f.checkpoint_dir, sharding_metadata, weights_only=True
    )
    manager = ocp.CheckpointManager(
        f.checkpoint_dir, options=ocp.CheckpointManagerOptions()
    )
    model.load_state(f.checkpoint_step, manager, weights_only=True)
    print("\nModel loaded!")


def create_app() -> FastAPI:
    app = FastAPI()

    @app.post("/gen_action")
    def gen_action(payload: dict[str, Any]) -> dict[str, Any]:
        _ensure_model_loaded()
        assert model is not None and config is not None

        obs_data = base64.b64decode(payload["obs"])
        obs = Image.open(BytesIO(obs_data))
        api_prompt = payload.get("prompt", "")

        if api_prompt != "":
            prompt = api_prompt
        else:
            prompt = input_prompt

        print(f"Prompt: {prompt}")

        start_time = time.time()
        action, viz = run_inference(model, prompt, obs, config)

        print(action)

        run_time = time.time() - start_time
        avg_time.append(run_time)
        print(f"Avg. run time: {np.array(avg_time).mean()}s")

        if viz is not None:
            viz_log = {k: wandb.Image(v) for k, v in viz.items()}
            run.log(viz_log)

        return {"action": action.tolist()}

    return app


if __name__ == "__main__":
    config_flags.DEFINE_config_file(
        "config", "configs/inference_config.py", "Path to the config file."
    )
    flags.DEFINE_string("platform", "gpu", "Platform to run on.")
    flags.DEFINE_string("checkpoint_dir", "", "Path to the checkpoint directory.")
    flags.DEFINE_integer("checkpoint_step", -1, "Step to resume from.")
    flags.DEFINE_string("prompt", "", "Prompt to generate action from.")
    flags.DEFINE_string("host", "0.0.0.0", "Host to bind.")
    flags.DEFINE_integer("port", 5000, "Port to bind.")

    flags.FLAGS(sys.argv)
    app = create_app()
    uvicorn.run(app, host=flags.FLAGS.host, port=flags.FLAGS.port)
