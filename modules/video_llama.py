from threading import Thread
import torch
from transformers import AutoModelForCausalLM
from transformers import AutoProcessor, TextIteratorStreamer, AutoConfig
from modules.base_model import BaseModel
from modules.model_factory import ModelFactory


@ModelFactory.register("videollama3-.*-image")
class VideoLLaMA(BaseModel):
    """Support VideoLLaMA3 with transformers library
    """

    def __init__(self, model_path, load_in_8bit=False, **kwargs) -> None:
        super().__init__()

        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        self.dtype = config.torch_dtype if hasattr(config, "torch_dtype") else torch.float16
        self.core_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=self.dtype,
            load_in_8bit=load_in_8bit,
            _attn_implementation="flash_attention_2" if self.supports_flash_attention else "eager"
        )
        self.core_model.eval()  # Set the model in eval mode explicitly even though not needed

        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    def support_image(self):
        return True

    def chatbot_to_messages(self, chatbot) -> list[str]:
        messages = []
        content = []
        for _, (user_msg, model_msg) in enumerate(chatbot):
            if isinstance(user_msg, (tuple, list)):  # query is image path
                content.append({"type": "image", "image": {"image_path": user_msg[0]}})
            else:  # query is text
                content.append({"type": "text", "text": user_msg})
                messages.append({"role": "user", "content": content})

        return messages

    def predict(self, chatbot, params):
        if len(chatbot) == 0 or not chatbot[-1][0] or chatbot[-1][1]:  # Empty user input or non-empty reply
            yield chatbot
        else:
            conversation = self.chatbot_to_messages(chatbot)

        inputs = self.processor(conversation=conversation, return_tensors="pt")
        inputs = {k: v.cuda(0) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)

        streamer = TextIteratorStreamer(self.processor.tokenizer, timeout=10,
                                        skip_prompt=True, skip_special_tokens=True)

        inputs["streamer"] = streamer
        inputs["max_new_tokens"] = 512
        t = Thread(target=self.core_model.generate, kwargs=inputs)
        t.start()

        for new_token in streamer:
            if self.stop_event.is_set():
                break

            if new_token != "":
                chatbot[-1][-1] += new_token
                yield chatbot

        t.join()
        self.stop_event.clear()

    @classmethod
    def description(cls) -> str:
        return "VideoLLaMA"
