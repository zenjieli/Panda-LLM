from utils.custom_config import CustomConfigs

model = None
custom_configs = CustomConfigs()

def model_invalidate_cache():
    if model is not None:
        model.invalidate_cache()