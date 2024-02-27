from enum import Enum


class ModelType(Enum):
    GPTQ = 'gptq'
    GGUF = 'gguf'
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

    # Check model path first
    if 'gptq' in model_path.lower():
        return ModelType.GPTQ
    elif 'gguf' in model_path.lower():
        return ModelType.GGUF
    else:
        # Check config
        try:
            config = AutoConfig.from_pretrained(model_path)
            if is_model_GPTQ(config):
                return ModelType.GPTQ
            else:
                return ModelType.Other
        except:
            return ModelType.Other
