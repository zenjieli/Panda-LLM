"""
Support MiniCPM with transformers library
"""
import torch
from PIL import Image
from typing import List
from modules.BaseModel import BaseModel
from transformers import AutoConfig, AutoModel, AutoTokenizer, BitsAndBytesConfig


class MiniCPMModel(BaseModel):
    _chat_completion_params = ["temperature", "top_p", "top_k", "repetition_penalty"]

    def __init__(self, model_path, load_in_8bit=False) -> None:
        super().__init__()

        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        dtype = config.torch_dtype if hasattr(config, "torch_dtype") else torch.float16
        kwargs = {"trust_remote_code": True,
                  "device_map": "cuda",
                  "torch_dtype": dtype}
        if hasattr(config, "version") and config.version == 2.6:
            kwargs["attn_implementation"] = "sdpa"

        if load_in_8bit:  # Check if load_in_8bit is needed
            if hasattr(config, "quantization_config") and \
                    (config.quantization_config.get("load_in_8bit", False) or config.quantization_config.get("load_in_4bit", False)):
                load_in_8bit = False

        if load_in_8bit:
            q_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_skip_modules=["out_proj", "kv_proj", "lm_head"])
            kwargs["quantization_config"] = q_config

        self.core_model = AutoModel.from_pretrained(model_path, **kwargs)
        self.core_model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    def dtype_in_config(config):
        # if torch_dtype exists in config, return it; otherwise, return float16
        return config.get("torch_dtype", torch.float16)

    def support_image(self):
        return True

    def chatbot_to_messages(self, chatbot, system_prompt) -> List[str]:
        messages = [{"role": "system", "content": system_prompt}] if system_prompt else []
        for _, (user_msg, model_msg) in enumerate(chatbot):
            if isinstance(user_msg, (tuple, list)):  # query is image path
                image = Image.open(user_msg[0]).convert("RGB")
            else:  # query is text
                messages.append({"role": "user", "content": user_msg})

        return messages, image

    def predict(self, chatbot, params):
        if len(chatbot) == 0 or not chatbot[-1][0] or chatbot[-1][1]:  # Empty user input or non-empty reply
            yield chatbot
        else:
            model_params, system_prompt, _ = BaseModel.gather_params(params, self._chat_completion_params)
            messages, image = self.chatbot_to_messages(chatbot, system_prompt)

            response = self.core_model.chat(
                image=image,
                msgs=messages,
                tokenizer=self.tokenizer,
                **model_params,
                stream=True)

            for item in response:
                if self.stop_event.is_set():
                    break

                if item:
                    chatbot[-1][-1] += item
                    yield chatbot

        self.stop_event.clear()
