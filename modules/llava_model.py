import torch
from PIL import Image
from modules.base_model import BaseModel
from modules.model_factory import ModelFactory

# Model ID must include "llava-onevision" or "llava-1.5" and end with "-hf"
@ModelFactory.register("llava-(onevision|1\.5|next).*?-hf$")
class LlavaModel(BaseModel):
    """
    Support LLaVA-OneVision, LLaVA-1.5 and LLaVA-Next with transformers library
    Tested with transformers==4.49.0
    """
    def __init__(self, model_path, **kwargs) -> None:
        from transformers import AutoProcessor
        if "1.5" in model_path:
            from transformers import LlavaForConditionalGeneration
            model_class = LlavaForConditionalGeneration
        elif "next" in model_path:
            from transformers import LlavaNextForConditionalGeneration
            model_class = LlavaNextForConditionalGeneration
        else:
            from transformers import LlavaOnevisionForConditionalGeneration, AutoProcessor
            model_class = LlavaOnevisionForConditionalGeneration

        super().__init__()
        self.core_model = model_class.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True).to(0)
        self.core_model.eval()  # Set the model in eval mode explicitly even though not needed
        self.processor = AutoProcessor.from_pretrained(model_path)

    def support_image(self):
        return True

    def chatbot_to_messages(self, chatbot) -> list[str]:
        messages = []
        content = []
        for _, (user_msg, model_msg) in enumerate(chatbot):
            if isinstance(user_msg, (tuple, list)):  # query is image path
                image = Image.open(user_msg[0])
                content.append({"type": "image"})
            else:  # query is text
                content.append({"type": "text", "text": user_msg})
                messages.append({"role": "user", "content": content})

        return messages, image

    def predict(self, chatbot, params):
        from time import time

        if len(chatbot) == 0 or not chatbot[-1][0] or chatbot[-1][1]:  # Empty user input or non-empty reply
            yield chatbot
        else:
            messages, image = self.chatbot_to_messages(chatbot)

            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = self.processor(images=[image], text=prompt, return_tensors='pt').to(0, torch.float16)
            t0 = time()
            output = self.core_model.generate(**inputs, max_new_tokens=200, do_sample=False)

            generated_ids = output[0][len(inputs.input_ids[0]):]
            token_count = generated_ids.shape[0]
            output_text = self.processor.decode(
                generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )

            chatbot[-1][-1] += output_text
            summary = f"New tokens: {token_count}; Speed: {token_count / (time() - t0):.1f} tokens/sec"
            yield chatbot, summary

    @classmethod
    def description(cls) -> str:
        return "LLaVA-1.5/Next/OneVision"