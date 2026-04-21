#!/bin/bash
echo "Launching TPU job..."
set -euo pipefail

VM_NAME="${1:?Usage: $0 <vm-name> <config-name>}"
CONFIG_NAME="${2:?Usage: $0 <vm-name> <config-name>}"
ZONE=us-central2-b

# Ensure SSH keys are provisioned on the VM (needed after VM recreation).
echo "Ensuring SSH keys are provisioned..."
gcloud compute tpus tpu-vm ssh "$VM_NAME" --zone="$ZONE" --command="echo ok" || {
    echo "ERROR: Cannot SSH into $VM_NAME. Is the TPU VM running?"
    exit 1
}

# Now get the SSH args for rsync via --dry-run.
SSH_CMD=$(gcloud compute tpus tpu-vm ssh "$VM_NAME" --zone="$ZONE" --dry-run 2>&1)
SSH_ARGS=$(echo "$SSH_CMD" | grep -oP '(?<=/usr/bin/ssh ).*(?= lovelace@)') || {
    echo "ERROR: Could not parse SSH args from dry-run output:"
    echo "$SSH_CMD"
    exit 1
}
TARGET_HOST=$(echo "$SSH_CMD" | grep -oP 'lovelace@[\d.]+') || {
    echo "ERROR: Could not parse target host from dry-run output:"
    echo "$SSH_CMD"
    exit 1
}

echo "Syncing to $TARGET_HOST ..."
rsync -avz --exclude='.venv' -e "/usr/bin/ssh ${SSH_ARGS//-t /}" \
/home/lovelace/CAST "$TARGET_HOST":/home/lovelace

gcloud compute tpus tpu-vm ssh "$VM_NAME" --zone="$ZONE" \
    --command="bash /home/lovelace/CAST/cast-vla/start_tpu_job.sh $WANDB_API_KEY $HF_TOKEN $CONFIG_NAME"
