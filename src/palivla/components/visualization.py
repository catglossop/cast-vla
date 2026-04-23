import numpy as np


def compute_action_mse(predicted_actions, gt_actions, actions_mask):
    """Compute total and per-dimension action MSE.

    Args:
        predicted_actions: (B, horizon, action_dim) predicted continuous actions.
        gt_actions: (B, horizon, action_dim) ground truth continuous actions.
        actions_mask: (B, horizon, action_dim) boolean mask for valid predictions.

    Returns:
        Dict with 'mse_total' and 'mse_dim_{i}' for each action dimension.
    """
    sq_error = np.square(predicted_actions - gt_actions)
    mask_sum = actions_mask.sum()
    if mask_sum == 0:
        action_dim = gt_actions.shape[-1]
        metrics = {"mse_total": float("nan")}
        for d in range(action_dim):
            metrics[f"mse_dim_{d}"] = float("nan")
        return metrics

    mse_total = np.sum(sq_error * actions_mask) / mask_sum

    action_dim = gt_actions.shape[-1]
    metrics = {"mse_total": float(mse_total)}
    for d in range(action_dim):
        dim_mask = actions_mask[..., d]
        dim_mask_sum = dim_mask.sum()
        if dim_mask_sum == 0:
            metrics[f"mse_dim_{d}"] = float("nan")
        else:
            metrics[f"mse_dim_{d}"] = float(
                np.sum(sq_error[..., d] * dim_mask) / dim_mask_sum
            )
    return metrics


def compute_action_accuracy(tokens_predicted, tokens_target, tokens_mask):
    """Compute token-level accuracy for action predictions.

    Args:
        tokens_predicted: (B, seq_len) predicted token IDs from autoregressive decode.
        tokens_target: (B, seq_len) ground truth token IDs.
        tokens_mask: (B, seq_len) mask indicating valid token positions.

    Returns:
        Dict with 'token_accuracy'.
    """
    mask_sum = tokens_mask.sum()
    if mask_sum == 0:
        return {"token_accuracy": float("nan")}

    correct = (tokens_predicted == tokens_target).astype(np.float32)
    accuracy = np.sum(correct * tokens_mask) / mask_sum
    return {"token_accuracy": float(accuracy)}


def run_eval_functions(predicted_actions, gt_actions, actions_mask, tokens):
    """Run all eval metric functions and return a combined dict.

    Args:
        predicted_actions: (B, horizon, action_dim) predicted continuous actions.
        gt_actions: (B, horizon, action_dim) ground truth continuous actions.
        actions_mask: (B, horizon, action_dim) boolean mask for valid predictions.
        tokens: Dict with keys 'predicted', 'target', 'mask' for token-level eval.

    Returns:
        Combined metrics dict.
    """
    metrics = {}
    metrics.update(compute_action_mse(predicted_actions, gt_actions, actions_mask))
    metrics.update(
        compute_action_accuracy(
            tokens["predicted"], tokens["target"], tokens["mask"]
        )
    )
    return metrics
