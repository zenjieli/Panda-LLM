"""
Support MiniCPM with transformers library
For version 2, use transformers <= 4.41.2. It is broken since 4.42.0 due to the default dynamic cache.
For version 2.5 and 2.6, use transformers >= 4.45.1
"""
import torch
from PIL import Image
from typing import Optional
import re
from modules.base_model import BaseModel
from transformers import AutoConfig, AutoModel, AutoTokenizer, BitsAndBytesConfig
from modules.model_factory import ModelFactory


@ModelFactory.register("minicpm")
class MiniCPMModel(BaseModel):
    _chat_completion_params = ["temperature", "repetition_penalty"]

    # Define version patterns: regex pattern -> (version, is_legacy)
    _version_patterns: list[tuple[re.Pattern, str, bool]] = [
        (re.compile(r"V?-?2[_-]5", re.I), "2.5", False),
        (re.compile(r"2[_-]6", re.I), "2.6", False),
        (re.compile(r"4[_-]5", re.I), "4.5", False),
        (re.compile(r"V?-?2$", re.I), "2", True),  # Matches "V-2" or ends with "2"
        (re.compile(r"MiniCPM-V$", re.I), "2", True),  # Explicit legacy model
    ]

    def __init__(self, model_path, load_in_8bit=False, **kwargs) -> None:
        super().__init__()

        version_info: Optional[tuple[str, bool]] = None

        for pattern, version, is_legacy in self._version_patterns:
            if pattern.search(model_path):
                version_info = (version, is_legacy)
                break

        if version_info is None:
            raise ValueError(f"Unsupported or unrecognized MiniCPM model path: {model_path}")

        self.version, self.is_legacy = version_info

        if self.is_legacy:
            self.core_model = MiniCPMModel.load_legacy_model(model_path)
        else:
            config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            kwargs = {"trust_remote_code": True,
                      "device_map": "cuda",
                      "torch_dtype": "auto"}
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
            print(self.core_model.dtype)
            self.core_model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    @staticmethod
    def load_legacy_model(model_path):
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16)
        model = model.to(device='cuda', dtype=torch.bfloat16)
        model.eval()
        return model

    def support_image(self):
        return True
    
    def support_video(self):
        return self.version == "4.5"

    def support_thinking(self):
        return self.version == "4.5"

    def chatbot_to_messages(self, chatbot, system_prompt) -> list[str]:
        messages = [{"role": "system", "content": system_prompt}] if system_prompt else []
        for _, (user_msg, model_msg) in enumerate(chatbot):
            if isinstance(user_msg, (tuple, list)):  # query is image path
                image = Image.open(user_msg[0]).convert("RGB")
            else:  # query is text
                messages.append({"role": "user", "content": user_msg})

        return messages, image

    def predict(self, chatbot, params):
        from collections.abc import Iterable

        if len(chatbot) == 0 or not chatbot[-1][0] or chatbot[-1][1]:  # Empty user input or non-empty reply
            yield chatbot
        else:
            model_params, system_prompt, _ = BaseModel.gather_params(params, self._chat_completion_params)
            messages, image = self.chatbot_to_messages(chatbot, system_prompt)
            
            if self.is_legacy:
                model_params["context"] = None

            response = self.core_model.chat(
                image=image,
                msgs=messages,
                tokenizer=self.tokenizer,
                **model_params,
                stream=not self.is_legacy)

            if self.is_legacy:
                chatbot[-1][-1] += response[0]
                yield chatbot
            else:
                for item in response:
                    if self.stop_event.is_set():
                        break

                    if item:
                        chatbot[-1][-1] += item
                        yield chatbot, ""

        self.stop_event.clear()

    @classmethod
    def description(cls) -> str:
        return "MiniCPM-V"
