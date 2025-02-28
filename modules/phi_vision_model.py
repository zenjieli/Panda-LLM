from PIL import Image
import torch
from transformers import AutoModelForCausalLM
from transformers import AutoProcessor, AutoConfig
from transformers import __version__ as transformers_version
from packaging import version
from modules.base_model import BaseModel
from modules.model_factory import ModelFactory

@ModelFactory.register("phi-.*-vision")
class PhiVisionModel(BaseModel):
    """Support PhiVisionModel with transformers library
    """

    def __init__(self, model_path, **kwargs) -> None:
        if version.parse(transformers_version) > version.parse("4.48.0"):
            raise Exception("PhiVisionModel requires transformers <= 4.48.0.")

        super().__init__()

        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        dtype = config.torch_dtype if hasattr(config, "torch_dtype") else torch.float16
        self.core_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=dtype,
            _attn_implementation="flash_attention_2" if self.supports_flash_attention else "eager"
        )
        self.core_model.eval()  # Set the model in eval mode explicitly even though not needed

        # For best performance, use num_crops=4 for multi-frame, num_crops=16 for single-frame.
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, num_crops=16)

    def support_image(self):
        return True

    def chatbot_to_messages(self, chatbot) -> tuple[list[str], Image.Image | list[Image.Image]]:
        messages = []
        images = []
        placeholder = ""
        for _, (user_msg, model_msg) in enumerate(chatbot):
            if isinstance(user_msg, (tuple, list)):  # query is image path
                image = Image.open(user_msg[0])
                images.append(image)
                placeholder += f"<|image_{len(images)}|>\n"
                # content.append({"type": "image"})
            else:  # query is text
                messages = [{"role": "user", "content": placeholder + user_msg}]

        return messages, images

    def predict(self, chatbot, params):
        if len(chatbot) == 0 or not chatbot[-1][0] or chatbot[-1][1]:  # Empty user input or non-empty reply
            yield chatbot
        else:
            messages, images = self.chatbot_to_messages(chatbot)

        prompt = self.processor.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.processor(prompt, images, return_tensors="pt").to("cuda")
        generation_args = {
            "max_new_tokens": 1000,
            "do_sample": False,
        }

        generate_ids = self.core_model.generate(**inputs,
                                                eos_token_id=self.processor.tokenizer.eos_token_id,
                                                **generation_args)

        # Remove input tokens
        generate_ids = generate_ids[:, inputs["input_ids"].shape[1]:]
        output_text = self.processor.batch_decode(generate_ids,
                                               skip_special_tokens=True,
                                               clean_up_tokenization_spaces=False)[0]

        chatbot[-1][-1] += output_text
        yield chatbot, ""

    @classmethod
    def description(cls) -> str:
        return "Phi Vision"