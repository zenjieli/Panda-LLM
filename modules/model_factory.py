import re

class ModelFactory:
    _creators = {}

    @classmethod
    def register(cls, model_name_pattern):
        def decorator(model_class):
            compiled_regex = re.compile(model_name_pattern, re.IGNORECASE)
            cls._creators[compiled_regex] = model_class
            return model_class
        return decorator

    @classmethod
    def get_model_class(cls, model_name: str) -> type|None:
        for regex, model_class in cls._creators.items():
            if regex.search(model_name):
                return model_class
        return None
