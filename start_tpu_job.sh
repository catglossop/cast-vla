#!/bin/bash
set -euo pipefail

cd /home/lovelace/CAST/cast-vla
export PATH="$HOME/.local/bin:$PATH"

# Install uv only if not already present.
if ! command -v uv &>/dev/null; then
    echo "Installing UV..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
else
    echo "UV already installed, skipping."
fi

# Only run uv sync if the venv doesn't exist or pyproject.toml is newer than the venv.
if [ ! -d .venv ] || [ pyproject.toml -nt .venv ]; then
    echo "Syncing virtual environment..."
    uv sync --extra tpu
else
    echo "Virtual environment up to date, skipping sync."
fi

WANDB_API_KEY="${1:?WANDB_API_KEY is required as first argument}"
HF_TOKEN="${2:?HF_TOKEN is required as second argument}"
CONFIG_NAME="${3:?CONFIG_NAME is required as third argument}"

source .venv/bin/activate

# Log in only if not already authenticated.
echo "Logging into Weights & Biases..."
uv run wandb login "$WANDB_API_KEY"

echo "Logging into Hugging Face..."
hf auth login --token "$HF_TOKEN" --add-to-git-credential

echo "Starting training..."
# Drop a forced backend from your laptop shell; on a real TPU VM JAX still picks TPU via jax[tpu].
unset JAX_PLATFORMS 2>/dev/null || true
# Optional: reduces XLA INTERNAL errors when FSDP collectives interact badly with
# certain slice fusions on TPU (override with your own LIBTPU_INIT_ARGS if needed).
export LIBTPU_INIT_ARGS="${LIBTPU_INIT_ARGS:-} --xla_tpu_enable_async_collective_fusion=false"
export PALIVLA_DEBUG_SHAPES=1

python scripts/train.py --config=$CONFIG_NAME --platform tpu --stats_only

