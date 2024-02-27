# Panda Webui

- [Panda Webui](#panda-webui)
  - [Overview](#overview)
  - [Installation](#installation)
  - [models](#models)
  - [Shell script](#shell-script)

## Overview

Panda WebUI is a lightweight and user-friendly web interface designed for running lightweight Language Models (LLMs) using the Hugging Face `transformers` library. In addition to supporting standard LLMs, Panda WebUI also accommodates GGUF models and several popular large Vision-Language Models (VLMs).

## Installation

* Create a new environment
```shell
conda create -n panda python=3.11
conda activate panda
```
* Install pytorch
```shell
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
```
* For `llama-cpp` (GGUF inference) and `deepspeed` (training):
```shell
conda install -y -c "nvidia/label/cuda-12.1.1" cuda
```
If training is not needed, one can install the CUDA Runtime instead:
```shell
conda install -y -c "nvidia/label/cuda-12.1.1" cuda-runtime
```
* Install the packages in requirements.txt
* Create a symbolic link `weights` to the model directory

## models

Set the Huggingface cache by setting the environment variable in `.bashrc`:
```
export HF_HOME=/path/to/cache
```


## Shell script

This shell script can be used to run from the terminal.

```shell
cd ~/path/to/Panda-LLM/ || { echo "Failed to change directory"; exit 1; }

# Activate the conda environment named 'panda'
source ~/miniforge3/etc/profile.d/conda.sh
conda activate panda

# Check if the conda environment 'panda' is active.
if [ "$CONDA_DEFAULT_ENV" = "panda" ]; then
    echo "Successfully activated conda environment 'panda'."
else
    echo "Failed to activate conda environment 'panda'."
    exit 1
fi

# Run the Python script
python webui.py
```