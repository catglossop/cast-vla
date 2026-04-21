import os
from typing import Any

import chex
import jax
import jax.numpy as jnp
import optax

from palivla.components.train_state import TrainState
from palivla import constants as c


def _palivla_host_shape_log(logits_shape, gen_shape, mask_shape):
    """Runs on host via jax.debug.callback; reads env here so it works without recompile."""
    if os.environ.get("PALIVLA_DEBUG_SHAPES", "").lower() not in ("1", "true", "yes"):
        return
    print(
        "[PALIVLA shapes] logits",
        logits_shape.tolist(),
        "gen.tokens",
        gen_shape.tolist(),
        "gen.mask_loss",
        mask_shape.tolist(),
        flush=True,
    )


def compute_stats_full_logits(
    *,
    logits: jax.Array,
    tokens: jax.Array,
    mask_loss: jax.Array,
):
    """Teacher-forcing loss with **no** sequence-axis slice on `logits`.

    ``logits[..., :-1, :]`` lowers to dynamic-slice on a large FSDP-sharded tensor and
    can trigger XLA INTERNAL (19 vs 20) on TPU.  Here ``logits`` stays ``[B, T, V]``;
    targets are shifted with ``concatenate`` on the **token** tensor; the last
    timestep’s CE is masked out (same objective as slice + ``tokens[..., 1:]``).
    """
    b = tokens.shape[0]
    z1 = jnp.zeros((b, 1), dtype=tokens.dtype)
    shift_labels = jnp.concatenate([tokens[..., 1:], z1], axis=-1)
    m = mask_loss.astype(jnp.float32)
    m_shift = jnp.concatenate([m[..., 1:], jnp.zeros((b, 1), dtype=jnp.float32)], axis=-1)
    ce = optax.softmax_cross_entropy_with_integer_labels(logits, shift_labels)
    denom = jnp.sum(m_shift) + jnp.finfo(jnp.float32).eps
    loss = jnp.sum(ce * m_shift) / denom
    pred = jnp.argmax(logits, axis=-1)
    accuracy = jnp.sum(m_shift * (pred == shift_labels).astype(jnp.float32)) / denom
    pred_valid = jnp.sum(m_shift * (pred > c.ACTION_TOKEN_START).astype(jnp.float32))
    valid_cnt = pred_valid / denom
    metrics = {"loss": loss, "accuracy": accuracy, "valid_cnt": valid_cnt}
    return loss, metrics


def compute_stats(
    *,
    pred_logits,
    target_tokens,
    target_mask_loss,
):
    loss = jnp.mean(
        target_mask_loss
        * optax.softmax_cross_entropy_with_integer_labels(pred_logits, target_tokens)
    ) / jnp.mean(target_mask_loss)
    accuracy = jnp.mean(
        target_mask_loss * (jnp.argmax(pred_logits, axis=-1) == target_tokens)
    ) / jnp.mean(target_mask_loss)
    
    pred_valid_tokens = jnp.count_nonzero(jnp.argmax(pred_logits, axis=-1) > c.ACTION_TOKEN_START)
    valid_cnt = pred_valid_tokens / pred_logits.shape[-2] 
    metrics = {"loss": loss, "accuracy": accuracy, "valid_cnt": valid_cnt}
    return loss, metrics


def step_fn(
    train_state: TrainState,
    batch: Any,
    key: chex.PRNGKey,
    train: bool,
):
    def loss_fn(params, batch, key: chex.PRNGKey):
        logits, _ = train_state.apply_fn(
            {"params": params},
            batch["sensors"],
            batch["sensors_mask"],
            batch["prompt"],
            batch["gen"],
            train=train,
        )

        # jax.debug.callback(
        #     _palivla_host_shape_log,
        #     jnp.asarray(logits.shape, dtype=jnp.int32),
        #     jnp.asarray(batch["gen"]["tokens"].shape, dtype=jnp.int32),
        #     jnp.asarray(batch["gen"]["mask_loss"].shape, dtype=jnp.int32),
        # )
        return compute_stats_full_logits(
            logits=logits,
            tokens=batch["gen"]["tokens"],
            mask_loss=batch["gen"]["mask_loss"],
        )
    grad_fn = jax.grad(loss_fn, has_aux=True)

    key, dropout_key = jax.random.split(key)
    grads, info = grad_fn(train_state.params, batch, dropout_key)
    train_state, info["optimizer"] = train_state.apply_gradients_with_info(grads=grads)

    return train_state, info, key
