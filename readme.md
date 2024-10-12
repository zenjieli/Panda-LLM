# Panda Webui

- [Panda Webui](#panda-webui)
  - [Installation](#installation)
  - [models](#models)
  - [Fine tuning](#fine-tuning)
  - [Shell script](#shell-script)



## Installation

* Create a new environment
```shell
conda create -n panda python=3.11
conda activate panda
```
* Install pytorch
```shell
pip install torch==2.2.1 torchvision==0.17.1 --index-url https://download.pytorch.org/whl/cu121
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
export TRANSFORMERS_CACHE=/path_to_cache
```



## Fine tuning

See [Supervised fine tuning](lora/readme.md).

## Shell script

This shell script can be used to run from the terminal.

```shell
cd ~/workspace/mine/Panda-LLM/ || { echo "Failed to change directory"; exit 1; }

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

