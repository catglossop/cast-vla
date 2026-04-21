#!/bin/bash
# After the job starts, opens local tmux session tpc_<pod-vm-name> via ssh_pod.sh (same as manual ssh_pod).
# Set LAUNCH_TPU_SKIP_MONITOR=1 to skip (e.g. non-interactive). Requires tmux and a TTY on stdout.
set -euo pipefail

VM_NAME="${1:?Usage: $0 <pod-vm-name> <config-name>}"
CONFIG_NAME="${2:?Usage: $0 <pod-vm-name> <config-name>}"
ZONE="${ZONE:-us-central2-b}"

: "${WANDB_API_KEY:?Export WANDB_API_KEY before running}"
: "${HF_TOKEN:?Export HF_TOKEN before running}"

command -v jq >/dev/null || {
    echo "ERROR: jq is required"
    exit 1
}

echo "Launching TPU pod job on $VM_NAME (zone=$ZONE)..."

TPU_JSON=$(gcloud alpha compute tpus tpu-vm describe "$VM_NAME" --zone="$ZONE" --format=json)
NUM_WORKERS=$(echo "$TPU_JSON" | jq -r '.networkEndpoints | length')
if [[ -z "$NUM_WORKERS" || "$NUM_WORKERS" == "0" ]]; then
    echo "ERROR: Could not read worker count from describe"
    exit 1
fi
echo "Detected $NUM_WORKERS worker(s)."

WORKER0_IP=$(echo "$TPU_JSON" | jq -r '.networkEndpoints[0].ipAddress')
if [[ -z "$WORKER0_IP" || "$WORKER0_IP" == "null" ]]; then
    echo "ERROR: Could not determine worker 0 IP address"
    exit 1
fi
COORDINATOR_PORT="${COORDINATOR_PORT:-12345}"
COORDINATOR_ADDR="${WORKER0_IP}:${COORDINATOR_PORT}"

echo "Coordinator: $COORDINATOR_ADDR"

echo "Ensuring SSH to worker 0..."
gcloud alpha compute tpus tpu-vm ssh "$VM_NAME" --zone="$ZONE" --worker=0 --command="echo ok" || {
    echo "ERROR: Cannot SSH into $VM_NAME worker 0"
    exit 1
}

rsync_one_worker() {
    local worker_id="$1"
    local SSH_CMD
    SSH_CMD=$(gcloud alpha compute tpus tpu-vm ssh "$VM_NAME" --zone="$ZONE" --worker="$worker_id" --dry-run 2>&1)
    local SSH_ARGS
    SSH_ARGS=$(echo "$SSH_CMD" | grep -oP '(?<=/usr/bin/ssh ).*(?= noam@)') || {
        echo "ERROR: Could not parse SSH args from dry-run (worker $worker_id)"
        echo "$SSH_CMD"
        return 1
    }
    local TARGET_HOST
    TARGET_HOST=$(echo "$SSH_CMD" | grep -oP 'noam@[\d.]+') || {
        echo "ERROR: Could not parse target host from dry-run (worker $worker_id)"
        echo "$SSH_CMD"
        return 1
    }
    echo "Syncing /hdd/CAST -> $TARGET_HOST (worker $worker_id)..."
    rsync -avz --exclude='.venv' -e "/usr/bin/ssh ${SSH_ARGS//-t /}" \
        /hdd/CAST "$TARGET_HOST":/home/noam
}

echo "Installing environment on all workers..."

# gcloud alpha compute tpus tpu-vm ssh "$VM_NAME" \
#   --zone="$ZONE" \
#   --worker=all \
#   --command="
# cd /home/noam/CAST/cast-vla
# export PATH=\$HOME/.local/bin:\$PATH

# if ! command -v uv &>/dev/null; then
#   curl -LsSf https://astral.sh/uv/install.sh | sh
# fi

# uv sync --extra tpu
# "

for ((w = 0; w < NUM_WORKERS; w++)); do
    rsync_one_worker "$w"
done

echo "Starting training on all workers (tmux session: tpc)..."

launch_one_worker() {
    local worker_id="$1"
    # Per-worker JAX multi-host coords (worker 0 runs the small coordinator server on COORDINATOR_PORT).
    local remote_cmd
    remote_cmd="export CAST_JAX_COORDINATOR_ADDRESS=$(printf '%q' "$COORDINATOR_ADDR"); export CAST_JAX_NUM_PROCESSES=$(printf '%q' "$NUM_WORKERS"); export CAST_JAX_PROCESS_ID=$(printf '%q' "$worker_id"); cd /home/noam/CAST/cast-vla && bash start_tpu_job.sh $(printf '%q' "$WANDB_API_KEY") $(printf '%q' "$HF_TOKEN") $(printf '%q' "$CONFIG_NAME")"

    gcloud alpha compute tpus tpu-vm ssh "$VM_NAME" \
        --zone="$ZONE" \
        --worker="$worker_id" \
        --command="tmux kill-session -t tpc 2>/dev/null || true; tmux new-session -d -s tpc bash -lc $(printf '%q' "$remote_cmd")"
}

for ((w = 0; w < NUM_WORKERS; w++)); do
    launch_one_worker "$w" &
done
wait

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ZONE

if [[ "${LAUNCH_TPU_SKIP_MONITOR:-0}" == "1" ]]; then
    echo "Skipping local tmux (LAUNCH_TPU_SKIP_MONITOR=1). Monitor with: ssh_pod.sh $VM_NAME $NUM_WORKERS"
    exit 0
fi

if ! command -v tmux >/dev/null 2>&1; then
    echo "tmux not found. Monitor with: ssh_pod.sh $VM_NAME $NUM_WORKERS"
    exit 0
fi

if [[ ! -t 1 ]]; then
    echo "stdout is not a TTY; skipping auto-attach. Monitor with: ssh_pod.sh $VM_NAME $NUM_WORKERS"
    exit 0
fi

echo "Opening local monitoring tmux: tpc_${VM_NAME} (${NUM_WORKERS} workers, zone=$ZONE)..."
bash "$SCRIPT_DIR/ssh_pod.sh" "$VM_NAME" "$NUM_WORKERS"