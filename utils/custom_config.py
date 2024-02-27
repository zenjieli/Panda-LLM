import json

class CustomConfig:
    def __init__(self, model_name, n_gpu_layers=-1, n_ctx_1024=1, lora_path=None, load_in_8bit=False, **kwargs):
        self.model_name = model_name
        self.n_gpu_layers = n_gpu_layers
        self.n_ctx_1024 = n_ctx_1024
        self.lora_path = lora_path
        self.load_in_8bit = load_in_8bit


class CustomConfigs:
    def __init__(self):
        import os.path as osp

        self.CONFIG_FILE_NAME = 'user_configs.json'
        self._configs = {}

        if osp.exists(self.CONFIG_FILE_NAME):
            self.load_from_json()

    def add(self, config: CustomConfig):
        """Add config if it doesn't exist; otherwise, update it
        """
        self._configs[config.model_name] = config

    def try_get(self, model_name):
        return self._configs.get(model_name, None)

    def save_to_json(self):
        data = {
            config.model_name: {
                'n_gpu_layers': config.n_gpu_layers,
                'n_ctx_1024': config.n_ctx_1024,
                'lora_path': config.lora_path,
                'load_in_8bit': config.load_in_8bit
            }
            for config in self._configs.values()
        }
        with open(self.CONFIG_FILE_NAME, 'w') as f:
            json.dump(data, f, indent=2)

    def load_from_json(self):
        with open(self.CONFIG_FILE_NAME, 'r') as f:
            data = json.load(f)
            for model_name, config_data in data.items():
                config = CustomConfig(model_name, **config_data)
                self.add(config)
