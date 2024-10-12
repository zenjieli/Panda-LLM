"""
Support QwenVL2Model with transformers library
"""
from PIL import Image
from modules.base_model import BaseModel
from modules.model_factory import ModelFactory


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
        content = []
        for _, (user_msg, model_msg) in enumerate(chatbot):
            if isinstance(user_msg, (tuple, list)):  # query is image path
                image = Image.open(user_msg[0]).convert("RGB")
                content.append({"type": "image"})
            else:  # query is text
                content.append({"type": "text", "text": user_msg})
                messages.append({"role": "user", "content": content})

        return messages, image

    def predict(self, chatbot, params):
        if len(chatbot) == 0 or not chatbot[-1][0] or chatbot[-1][1]:  # Empty user input or non-empty reply
            yield chatbot
        else:
            messages, image = self.chatbot_to_messages(chatbot)

            # Preprocess the inputs
            text_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            # Excepted output: '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe this image.<|im_end|>\n<|im_start|>assistant\n'

            inputs = self.processor(text=[text_prompt], images=[image], padding=True, return_tensors="pt")
            inputs = inputs.to("cuda")

            # Inference: Generation of the output
            output_ids = self.core_model.generate(**inputs, max_new_tokens=128)
            generated_ids = [
                output_ids[len(input_ids):]
                for input_ids, output_ids in zip(inputs.input_ids, output_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )

            chatbot[-1][-1] += "\n".join(output_text)
            yield chatbot

    @classmethod
    def description(cls) -> str:
        return "Qwen2-VL"