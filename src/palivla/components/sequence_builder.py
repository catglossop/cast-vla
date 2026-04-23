from dataclasses import dataclass
from os import PathLike

import cloudpickle
import einops
import numpy as np
import tensorflow as tf
from transformers import AutoTokenizer

from big_vision.utils import Registry
from palivla.components.action_tokenizer import ActionTokenizer


@Registry.register("sequence_builder.default")
@dataclass
class SequenceBuilder:
    prompt_pad_length: int
    gen_pad_length: int

    def __post_init__(self):
        self.prompt_pad_length = self.prompt_pad_length
        self.gen_pad_length = self.gen_pad_length

    def save(self, path: PathLike):
        with tf.io.gfile.GFile(tf.io.gfile.join(path, "sequence_builder.pkl"), "wb") as f:
            cloudpickle.dump(self, f)

    @classmethod
    def load(cls, path: PathLike):
        with tf.io.gfile.GFile(tf.io.gfile.join(path, "sequence_builder.pkl"), "rb") as f:
            return cloudpickle.load(f)

    def prepare_prompt(self, language_instruction): 
        if isinstance(language_instruction, bytes):
            language_instruction = [language_instruction]
        language_instructions = np.array([l.decode("utf-8") for l in language_instruction])
        language_instructions = language_instructions[language_instructions != ""]
        if language_instructions.shape[0] != 0:
            lang = np.random.choice(language_instructions)
            return "<bos>" + str(lang)
        else:
            return "<bos>" + str("")

    def prepare_gen(self, action_tokens):
        return "".join([f"<act{i}>" for i in action_tokens]) + "<eos>"

    def _batch_pad(self, token_lists, pad_length, pad_value=0):
        """Pad a list of variable-length 1-D arrays into a single (B, pad_length) array."""
        batch_size = len(token_lists)
        out = np.full((batch_size, pad_length), pad_value, dtype=np.int32)
        for i, tok in enumerate(token_lists):
            length = min(len(tok), pad_length)
            out[i, :length] = tok[:length]
        return out

    def build_sequence(
        self,
        batch,
        language_tokenizer: AutoTokenizer,
        action_tokenizer: ActionTokenizer,
        boa_is_prompt: bool = False,
        include_action_tokens: bool = True,
    ):
        boa_id = language_tokenizer.encode("<begin_of_action>")[0]
        boa_prompt = "<begin_of_action>" if boa_is_prompt else ""
        boa_gen = "" if boa_is_prompt else "<begin_of_action>"

        prompt = [
            self.prepare_prompt(instruction) + boa_prompt
            for instruction in batch["task"]["language_instruction"]
        ]

        prompt_tokens = language_tokenizer(prompt)["input_ids"]

        if include_action_tokens:
            action_tokens = [
                boa_gen + self.prepare_gen(t)
                for t in action_tokenizer.tokenize(batch["action"][..., -1, :, :])
            ]
            action_tokens = language_tokenizer(action_tokens)[
                "input_ids"
            ]
        else:
            action_tokens = [[] for _ in range(len(prompt_tokens))]

        batch_size = len(prompt_tokens)
        lengths_prompt = np.array([min(len(t), self.prompt_pad_length) for t in prompt_tokens])
        lengths_gen = np.array([min(len(t), self.gen_pad_length) for t in action_tokens])

        prompt_tok_arr = self._batch_pad(prompt_tokens, self.prompt_pad_length)
        gen_tok_arr = self._batch_pad(action_tokens, self.gen_pad_length)

        prompt_mask = np.arange(self.prompt_pad_length)[None, :] < lengths_prompt[:, None]
        gen_mask = np.arange(self.gen_pad_length)[None, :] < lengths_gen[:, None]

        return {
            "prompt": {
                "tokens": prompt_tok_arr,
                "mask": prompt_mask,
                "mask_ar": prompt_tok_arr == boa_id,
                "mask_loss": np.zeros((batch_size, self.prompt_pad_length), dtype=bool),
            },
            "gen": {
                "tokens": gen_tok_arr,
                "mask": gen_mask,
                "mask_ar": np.ones((batch_size, self.gen_pad_length), dtype=bool),
                "mask_loss": gen_mask,
            },
        }

    def get_actions(
        self,
        tokens: np.ndarray,
        language_tokenizer: AutoTokenizer,
        action_tokenizer: ActionTokenizer,
        *,
        boa_is_prompt: bool = False,
        action_dim: int | None,
        boa_id: int | None = None,
        eos_id: int | None = None,
        act0_id: int | None = None,
    ):
        boa_id = boa_id or language_tokenizer.encode("<begin_of_action>")[0]
        eos_id = eos_id or language_tokenizer.encode("<eos>")[0]
        act0_id = act0_id or language_tokenizer.encode("<act0>")[0]
        # Find the beginning of the action
        if boa_is_prompt:
            start_idx = 0
        else:
            try:
                start_idx = np.where(tokens == boa_id)[0][0] + 1
            except IndexError:
                return None

        # Find the end of the action
        try:
            end_idx = np.where(tokens == eos_id)[0][0]
        except IndexError:
            return None
        
        # Get the action
        action = tokens[start_idx:end_idx] - act0_id
        try:
            return action_tokenizer.detokenize(action, action_dim=action_dim)
        except ValueError:
            return None

    def batch_get_actions(
        self,
        tokens,
        language_tokenizer: AutoTokenizer,
        action_tokenizer: ActionTokenizer,
        *,
        boa_is_prompt: bool = False,
        action_dim: int,
        action_horizon: int,
    ):
        boa_id = language_tokenizer.encode("<begin_of_action>")[0]
        eos_id = language_tokenizer.encode("<eos>")[0]
        act0_id = language_tokenizer.encode("<act0>")[0]
        
        actions = [
            self.get_actions(
                tokens[i],
                language_tokenizer,
                action_tokenizer,
                boa_is_prompt=boa_is_prompt,
                boa_id=boa_id,
                eos_id=eos_id,
                act0_id=act0_id,
                action_dim=action_dim,
            )
            for i in range(len(tokens))
        ]
        actions = [actions[i].squeeze(0) if actions[i] is not None and len(actions[i].shape) == 3 else actions[i] for i in range(len(actions))]

        actions_mask = np.array([action is not None for action in actions])
        try:
            actions = np.stack(
                [
                    (
                        np.pad(
                            action,
                            ((0, action_horizon - action.shape[0]), (0, 0)),
                            constant_values=np.nan,
                        )
                        if action is not None
                        else np.zeros((action_horizon, action_dim))
                    )
                    for action in actions
                ]
            )
        except Exception as e:
            actions = np.zeros((len(actions), action_horizon, action_dim))
        actions_mask = einops.repeat(
            actions_mask, "b -> b p a", p=action_horizon, a=action_dim
        ) & ~np.isnan(actions)

        return actions, actions_mask
