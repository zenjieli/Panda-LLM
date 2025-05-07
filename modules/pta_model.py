from PIL import Image
import torch
from transformers import AutoModelForCausalLM
from transformers import AutoProcessor, AutoConfig
from transformers import __version__ as transformers_version
from packaging import version
from modules.base_model import BaseModel
from modules.model_factory import ModelFactory


@ModelFactory.register("PTA-\d+(\.\d+)?")
class PTAModel(BaseModel):
    """Support AskUI/PTA model with transformers library
    """

    def __init__(self, model_path, **kwargs) -> None:
        super().__init__()

        self.core_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
            _attn_implementation="flash_attention_2" if self.supports_flash_attention else "eager"
        )
        self.core_model.eval()  # Set the model in eval mode explicitly even though not needed
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    def support_image(self):
        return True

    def chatbot_to_messages(self, chatbot) -> tuple[list[str], Image.Image | list[Image.Image]]:
        for _, (user_msg, model_msg) in enumerate(chatbot):
            if isinstance(user_msg, (tuple, list)):  # query is image path
                image = Image.open(user_msg[0]).convert("RGB")
            else:  # query is text
                message = user_msg

        return message, image

    def predict_simple_nostream(self, query: str, system: str, gen_kwargs: dict) -> str:
        """Simply predict without streaming. Useful for OpenAI API server.
        """
        image_url = gen_kwargs.get("image_url", None)
        image = Image.open(image_url).convert("RGB") if image_url else None

        task_prompt = "<OPEN_VOCABULARY_DETECTION>"
        prompt = task_prompt + query

        inputs = self.processor(prompt, image, return_tensors="pt").to("cuda", self.core_model.dtype)
        generation_args = {
            "max_new_tokens": 1024,
            "do_sample": False,
            "num_beams": 3
        }

        generated_ids = self.core_model.generate(**inputs,
                                                 eos_token_id=self.processor.tokenizer.eos_token_id,
                                                 **generation_args)

        # Remove input tokens
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = self.processor.post_process_generation(generated_text, task="<OPEN_VOCABULARY_DETECTION>",
                                                               image_size=(image.width, image.height))

        polygon = parsed_answer[task_prompt]["polygons"][0]
        if isinstance(polygon, list):
            polygon = polygon[0]

        polygon_points_str = []
        for i in range(0, len(polygon), 2):
            polygon_points_str.append(f"({round(polygon[i])}, {round(polygon[i+1])})")

        return f"Grounded UI element polygon: {', '.join(polygon_points_str)}"

    def predict(self, chatbot, params):
        if len(chatbot) == 0 or not chatbot[-1][0] or chatbot[-1][1]:  # Empty user input or non-empty reply
            yield chatbot
        else:
            message, image = self.chatbot_to_messages(chatbot)

        # The message is like "description of the target element <UI element name>"
        # See huggingface.co/AskUI/PTA-1.0/
        task_prompt = "<OPEN_VOCABULARY_DETECTION>"
        prompt = task_prompt + message

        inputs = self.processor(prompt, image, return_tensors="pt").to("cuda", self.core_model.dtype)
        generation_args = {
            "max_new_tokens": 1024,
            "do_sample": False,
            "num_beams": 3
        }

        generated_ids = self.core_model.generate(**inputs,
                                                 eos_token_id=self.processor.tokenizer.eos_token_id,
                                                 **generation_args)

        # Remove input tokens
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = self.processor.post_process_generation(generated_text, task="<OPEN_VOCABULARY_DETECTION>",
                                                               image_size=(image.width, image.height))

        polygon = parsed_answer[task_prompt]["polygons"][0]
        if isinstance(polygon, list):
            polygon = polygon[0]

        polygon_points_str = []
        for i in range(0, len(polygon), 2):
            polygon_points_str.append(f"({round(polygon[i])}, {round(polygon[i+1])})")

        chatbot[-1][-1] += f"Grounded region: {', '.join(polygon_points_str)}"
        yield chatbot, ""

    @classmethod
    def description(cls) -> str:
        return "Phi Vision"
