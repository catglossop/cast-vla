#!/bin/bash
# Activate ssh key 
SSH_KEY_NAME=$1
mkdir -p /home/$USER/.ssh
mv $SSH_KEY_NAME /home/$USER/.ssh
eval "$(ssh-agent -s)"
ssh-add $SSH_KEY_NAME
touch /home/$USER/.ssh/known_hosts
ssh-keyscan github.com >> /home/$USER/.ssh/known_hosts
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env
git clone https://github.com/catglossop/cast-vla.git --recursive
cd ~/cast-vla
git pull
git submodule sync --recursive
cd ~/cast-vla/octo
git fetch 
git checkout origin/main
git branch main -f 
git checkout main
cd ~/cast-vla
source .venv/bin/activate
uv python pin 3.11.12
uv venv --python=python3.11.12
uv sync --extra tpu  

# For inference
uv pip install google-cloud-logging
uv pip install google-cloud-storage