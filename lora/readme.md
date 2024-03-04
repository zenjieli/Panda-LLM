- [Supervised fine tuning](#supervised-fine-tuning)
  - [QWen 1.5](#qwen-15)
    - [Prerequisites](#prerequisites)
    - [LORA fine-tuning](#lora-fine-tuning)
    - [Merge LORA and original files](#merge-lora-and-original-files)


# Supervised fine tuning

## QWen 1.5

### Prerequisites

```shell
conda install -y -c "nvidia/label/cuda-12.1.1" cuda
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.3.4/flash_attn-2.3.4+cu122torch2.1cxx11abiFALSE-cp311-cp311-linux_x86_64.whl; platform_system == "Linux" and platform_machine == "x86_64" and python_version == "3.11"
pip install peft deepspeed optimum accelerate
```

See `Text Generation Webui` readme for the latest `flash-attention` package.

### LORA fine-tuning

1. Clone <https://github.com/QwenLM/Qwen1.5>.
2. `cd Qwen1.5/examples/sft`
3. Run
   ```shell
   CUDA_VISIBLE_DEVICES="0" bash finetune.sh -m <model_path> -d <data_path> --deepspeed <config_path> --use_lora True
   ```
   where `<config_path>` can be `ds_config_zero2.json` or `ds_config_zero3.json`.

### Merge LORA and original files

```python
from peft import AutoPeftModelForCausalLM

model = AutoPeftModelForCausalLM.from_pretrained(
    path_to_adapter_dir,
    device_map="auto"
).eval()

merged_model = model.merge_and_unload()
merged_model.save_pretrained(new_model_directory)

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(path_to_adapter_dir, trust_remote_code=True)
tokenizer.save_pretrained(new_model_directory)
```

**References**
* <https://qwen.readthedocs.io/>
* <https://qwen.readthedocs.io/en/latest/training/SFT/example.html>.