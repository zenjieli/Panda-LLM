from threading import Event, Thread
from typing import Generator
import torch
from transformers import TextIteratorStreamer
import utils.text_processing as text_processing


class BaseModel:
    def __init__(self) -> None:
        self.stop_event = Event()
        self.core_model = None
        self.supports_flash_attention = self.__supports_flash_attention()

    def predict(self):
        pass

    @staticmethod
    def __supports_flash_attention():
        """Check if a GPU supports FlashAttention.
        """
        for device_id in range(torch.cuda.device_count()):
            major, _ = torch.cuda.get_device_capability(device_id)
            # Returns false unless the GPU architecture is Ampere (SM 8.x) or newer (SM 9.0)
            if major < 8:
                return False

        return True

    def support_image(self):
        return False

    def support_video(self):
        return False

    def append_user_input(self, query: str, chatbot: list[list]) -> tuple[str, list[list]]:
        if chatbot is None:
            chatbot = []

        if query != '':
            return '', chatbot + [[text_processing.parse_text(query), '']]
        else:
            return '', chatbot

    def try_tokenize(self, chatbot, system_prompt) -> list:
        return []

    def check_token_count(self, token_count: int) -> bool:
        """Returns true if the current token count is OK for inference
        """
        return True

    def chatbot_to_messages(chatbot, system_prompt) -> list[str]:
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

    def gather_params(user_param_elements, expected_params) -> tuple[dict, str, bool]:
        """Gather model parameters
        Parameters:
            user_param_elements: user input elements
            expected_params: model parameters expected by the model
        """
        model_params = {params_name: user_param_elements[params_name]
                        for params_name in expected_params if params_name in user_param_elements}
        enable_postprocessing = user_param_elements.get('enable_postprocessing', False)
        return model_params, user_param_elements.get('system_prompt'), enable_postprocessing

    def num_params(self) -> int:
        if hasattr(self.core_model, "parameters"):
            return sum(p.numel() for p in self.core_model.parameters())
        else:  # Not supported
            return 0

    def generate_stream(self, streamer: TextIteratorStreamer, inputs: dict)->Generator[str, None, None]:
        t = Thread(target=self.core_model.generate, kwargs=inputs)
        t.start()

        for new_token in streamer:
            if self.stop_event.is_set():
                break

            yield new_token

        t.join()
        self.stop_event.clear()

    def get_meta_info(self) -> dict:
        return ""

    @classmethod
    def description(self) -> str:
        return ""

    @staticmethod
    def is_image_file(filename: str)->bool:
        ext = "." in filename and filename.split(".")[-1].lower()
        return ext in ["jpg", "jpeg", "png", "gif"]

    @staticmethod
    def is_video_file(filename: str)->bool:
        ext = "." in filename and filename.split(".")[-1].lower()
        return ext == "mp4"

