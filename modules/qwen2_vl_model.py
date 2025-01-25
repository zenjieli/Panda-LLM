"""
Support QwenVL2Model with transformers library
"""
from threading import Thread
from PIL import Image
from modules.base_model import BaseModel
from modules.model_factory import ModelFactory
from transformers import TextIteratorStreamer


@ModelFactory.register("qwen2-vl")
class Qwen2VLModel(BaseModel):
    def __init__(self, model_path, **kwargs) -> None:
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

        super().__init__()

        self.core_model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype="auto",
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
        self.core_model.eval()  # Set the model in eval mode explicitly even though not needed

        self.processor = AutoProcessor.from_pretrained(model_path)

    def support_image(self):
        return True

    def chatbot_to_messages(self, chatbot) -> list[str]:
        messages = []
        image = None
        content = []
        for _, (user_msg, model_msg) in enumerate(chatbot):
            if isinstance(user_msg, (tuple, list)):  # query is image path
                image = Image.open(user_msg[0]).convert("RGB")
                content.append({"type": "image"})
            else:  # query is text
                if user_msg:
                    content.append({"type": "text", "text": user_msg})
                    messages.append({"role": "user", "content": content})
                    content = []
                if model_msg:
                    messages.append({'role': 'assistant', 'content': model_msg})

        return messages, image

    def predict(self, chatbot: list[list[str | tuple]], params: dict):
        from time import time

        if len(chatbot) == 0 or not chatbot[-1][0] or chatbot[-1][1]:  # Empty user input or non-empty reply
            yield chatbot
        else:
            messages, image = self.chatbot_to_messages(chatbot)

            # Preprocess the inputs
            text_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            # Excepted output: '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe this image.<|im_end|>\n<|im_start|>assistant\n'

            inputs = self.processor(text=[text_prompt], images=[image]
                                    if image is not None else None, padding=True, return_tensors="pt")
            inputs = inputs.to("cuda")

            streamer = TextIteratorStreamer(self.processor.tokenizer, timeout=10,
                                            skip_prompt=True, skip_special_tokens=True)

            inputs["streamer"] = streamer
            inputs["max_new_tokens"] = 512
            token_count = 0
            t0 = time()
            for new_token in self.generate_stream(streamer, inputs):
                token_count += 1
                if new_token != "":
                    chatbot[-1][-1] += new_token
                    yield chatbot, ""

            summary = f"New tokens: {token_count}; Speed: {token_count / (time() - t0):.1f} tokens/sec"
            yield chatbot, summary

    @classmethod
    def description(cls) -> str:
        return "Qwen2-VL"
