class OpenaiAPIModel:
    def __init__(self, model_id, model_name):
        self.core_model = None

class OpenaiAPIGGUFModel(OpenaiAPIModel):
    """For llama-cpp GGUF models
    """
    def __init__(self, model_id, model_name):
        super().__init__(model_id, model_name)

class OpenaiAPIAutoModel(OpenaiAPIModel):
    """For transformers auto models
    """
    def __init__(self, model_id, model_name):
        super().__init__(model_id, model_name)