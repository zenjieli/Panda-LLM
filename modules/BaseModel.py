from threading import Event
from typing import List, Tuple
import utils.text_processing as text_processing


class BaseModel:
    def __init__(self) -> None:
        self.stop_event = Event()
        self.core_model = None

    def predict(self):
        pass

    def support_image(self):
        return False

    def append_user_input(self, query: str, chatbot: List[List]) -> Tuple[str, List[List]]:
        if chatbot is None:
            chatbot = []

        if query != '':
            return '', chatbot + [[text_processing.parse_text(query), '']]
        else:
            return '', chatbot

    def try_tokenize(self, chatbot, system_prompt) -> List:
        return []

    def check_token_count(self, token_count: int) -> bool:
        """Returns true if the current token count is OK for inference
        """
        return True

    def chatbot_to_messages(chatbot, system_prompt) -> List[str]:
        messages = [{'role': 'system', 'content': system_prompt}] if system_prompt else []
        for idx, (user_msg, model_msg) in enumerate(chatbot):
            if idx == len(chatbot) - 1 and not model_msg:
                messages.append({'role': 'user', 'content': user_msg})
                break
            if user_msg:
                messages.append({'role': 'user', 'content': user_msg})
            if model_msg:
                messages.append({'role': 'assistant', 'content': model_msg})

        return messages

    def tokenizer(self):
        return None

    def gather_params(user_param_elements, expected_params) -> Tuple[dict, str, bool]:
        """Gather model parameters
        Parameters:
            user_param_elements: user input elements
            expected_params: model parameters expected by the model
        """
        model_params = {params_name: user_param_elements[params_name]
                        for params_name in expected_params if params_name in user_param_elements}
        enable_postprocessing = user_param_elements.get('enable_postprocessing', False)
        return model_params, user_param_elements.get('system_prompt'), enable_postprocessing
