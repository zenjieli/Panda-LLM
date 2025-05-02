"""
Support Janus with transformers library
"""
from modules.base_model import BaseModel
from modules.model_factory import ModelFactory
from transformers import AutoModelForCausalLM, AutoProcessor
from transformers import __version__ as transformers_version
from packaging import version
import torch


@ModelFactory.register("janus")
class JanusModel(BaseModel):
    def __init__(self, model_path, **kwargs) -> None:
        super().__init__()

        if version.parse(transformers_version) >= version.parse("4.49.0"):
            try:
                from janus.models import MultiModalityCausalLM, VLChatProcessor
                from janus.utils.io import load_pil_images
            except ImportError as e:
                print("Failed to import janus:")
                raise
        else:
            raise RuntimeError("Error: Janus requires transformers >= 4.49.0.")

        self.vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
        self.tokenizer = self.vl_chat_processor.tokenizer

        self.core_model: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True)
        self.core_model = self.core_model.to(torch.bfloat16).cuda()
        self.core_model.eval()  # Set the model in eval mode explicitly even though not needed

        self.processor = AutoProcessor.from_pretrained(model_path)

    def support_image(self):
        return True

    def chatbot_to_messages(self, chatbot) -> list[str]:
        messages = []
        images = []
        for user_msg, model_msg in chatbot:
            if isinstance(user_msg, (tuple, list)):  # query is image path
                if BaseModel.is_image_file(user_msg[0]):
                    images.append(user_msg[0])
                else:
                    raise ValueError(f"Unsupported file type: {user_msg[0]}")
            elif isinstance(user_msg, str):  # query is text
                if user_msg:
                    text_content = {"role": "<|User|>",
                                    "content": f"<image_placeholder>\n{user_msg}",
                                    "images": images}
                    messages.append(text_content)
                    images = []
                if model_msg:
                    messages.append({"role": "<|Assistant|>", "content": ""})

        return messages

    def predict(self, chatbot: list[list[str | tuple]], params: dict):
        from time import time

        if len(chatbot) == 0 or not chatbot[-1][0] or chatbot[-1][1]:  # Empty user input or non-empty reply
            yield chatbot
        else:
            messages = self.chatbot_to_messages(chatbot)

            t0 = time()
            # Load images and prepare for inputs
            pil_images = load_pil_images(messages)
            prepare_inputs = self.vl_chat_processor(
                conversations=messages, images=pil_images, force_batchify=True
            ).to(self.core_model.device)

            # # run image encoder to get the image embeddings
            inputs_embeds = self.core_model.prepare_inputs_embeds(**prepare_inputs)

            # # run the model to get the response
            outputs = self.core_model.language_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=prepare_inputs.attention_mask,
                pad_token_id=self.tokenizer.eos_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=512,
                do_sample=False,
                use_cache=True)

            chatbot[-1][-1] = self.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
            token_count = len(outputs[0])
            summary = f"New tokens: {token_count}; Speed: {token_count / (time() - t0):.1f} tokens/sec"
            yield chatbot, summary

    @classmethod
    def description(cls) -> str:
        return "Janus"
