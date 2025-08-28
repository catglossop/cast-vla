# Quickstart for CAST finetuning

Note that the script to run on tpus will automatically push your code. If you want to isolate your development please make a new branch. Read the launch script `run_tpu.sh` carefully before launching.

To launch on a single tpu vm (v4-8)
```bash
bash run_tpu.sh <name of tpu> <initialize (true for first job on new tpu)> <update (true if code is changed)> <wandb api key> <config-name>
```
To ssh into a single tpu vm (v4-8)
```bash
gcloud alpha compute tpus tpu-vm ssh <name of tpu> --zone=us-central2-b
```
To launch on a pod
```bash
bash run_tpu_pod.sh <name of pod> <initialize (true for first job on new tpu)> <update (true if code is changed)> <wandb api key> <config-name>
```
To ssh into a pod 
```bash
bash ssh_pod.sh <name of pod>
```

Note that on initialization of a new pod or tpu vm, you will need to login to hugging face (to be fixed). To do this, ssh into the single vm or pod,
```bash
gcloud compute tpus tpu-vm ssh <name-of-tpu-vm> --zone=<your-region>
source ~/cast-vla/.venv/bin/activate
huggingface-cli login
```
and input your token. 

To run inference, 
```bash
python inference_server.py --platform <gpu or tpu> --checkpoint_dir <your checkpoint dir> --checkpoint_step <your checkpoint step> --prompt <the prompt to the model>
```

# PaliVLA
This is a framework for training multimodal vision-language-action (VLA) model for robotics in JAX. It primarily supports PaliGemma for now, though more base models will be added in the future.

## Installation
We develop with `uv`, but other environment managers should work fine. To install the dependencies, run:
```bash
uv venv
uv sync
```

## Training
To train a model, run:
```bash
python scripts/train.py --config <your config name>
```

This repository is (for now) a fork of [`big_vision`](https://github.com/google-research/big_vision).

## Citation
If you use PaliVLA in your own project, please cite this repository:
```bibtex
@misc{palivla,
  author       = {Kyle Stachowicz},
  title        = {PaliVLA},
  year         = {2024},
  url          = {https://github.com/kylestach/bigvision-palivla},
  note         = {GitHub repository}
}
```
