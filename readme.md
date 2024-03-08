# Panda Webui

- [Panda Webui](#panda-webui)
  - [Installation](#installation)
  - [models](#models)
    - [Yi](#yi)
  - [Fine tuning](#fine-tuning)
  - [Known issues](#known-issues)



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
* Create a soft link to the model directory

## models

### Yi

There is bug in llama.cpp in converting Yi models to `gguf`. See the fix in <https://github.com/01-ai/Yi/blob/main/docs/README_llama.cpp.md>. When using `gguf` models, one should use `chatml` prompt template. A fixed GGUF can be downloaded from onedrive.

**Note**:
* Yi-34B-34bits will crash the system and files
* Generally, Yi-34B--GGUF models suffer a lot from illusions and non-stopping replies



## Fine tuning

See [Supervised fine tuning](lora/readme.md).

## Known issues

* Quantization
  * The AWQ model does not work properly. It never ends.
  * Only GGUF works with multiple GPUs
* Model specific
  * Mistral doesn't accept the system prompt
  * Qwen14B-Chat-Int8: runtime error: probability tensor contains either `inf`, `nan` or element < 0
* Hardware specific
  * `Cuda extension not installed` error when loading GPTQ models with 4060Ti GPUs
