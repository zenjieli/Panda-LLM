from enum import Enum


class ModelType(Enum):
    GGUF = 'gguf'
    GPTQ = 'gptq'
    InternVL2 = 'InternVL2'
    LLaVA = 'llava.*-gguf'
    LLaVA_OneVision = 'llava-onevision-qwen2'
    MiniCPM = 'minicpm'
    PhiVision = 'phi-.*-vision'
    QWEN_VL = 'qwen-vl'
    QWEN2_VL = 'qwen2-vl'
    Other = 'other'


def is_model_GPTQ(config):
    quantization_config = getattr(config, 'quantization_config', None)

    if quantization_config:
        quant_method = quantization_config.get('quant_method', None)
        return quant_method == 'gptq'
    else:
        return False


def get_model_type(model_path):
    from transformers import AutoConfig
    import re

    # Check model path first
    for model_type in ModelType:
        if model_type != ModelType.Other and re.search(model_type.value, model_path, re.IGNORECASE):
            return model_type

    # Check config
    try:
        config = AutoConfig.from_pretrained(model_path)
        if is_model_GPTQ(config):
            return ModelType.GPTQ
        else:
            return ModelType.Other
    except:
        return ModelType.Other

def get_block_count_from_llama_meta(llama_meta: dict) -> int:
    # Find the key ending with '.block_count'
    for key in llama_meta.keys():
        if key.endswith('.block_count'):
            return int(llama_meta[key])

    return -1