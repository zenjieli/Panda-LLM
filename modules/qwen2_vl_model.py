from modules.base_model import BaseModel
from modules.model_factory import ModelFactory
from transformers import TextIteratorStreamer, AutoProcessor, AutoModel
from transformers.utils import check_min_version
from qwen_vl_utils import process_vision_info


@ModelFactory.register("qwen2(\.5)?-vl|UI-TARS-1\.5")
class Qwen2VLModel(BaseModel):
    """
    Support QwenVL2Model with transformers library
    Tested with transformers==4.49.0    
    """

    def __init__(self, model_path, load_in_8bit=False, **kwargs) -> None:
        super().__init__()

        import re
        version_2_5 = re.search("qwen2\.5-vl|UI-TARS-1\.5", model_path, re.IGNORECASE)

        if version_2_5:
            check_min_version("4.49.0.dev0")
            from transformers import Qwen2_5_VLForConditionalGeneration
            model_class = Qwen2_5_VLForConditionalGeneration
        else:
            from transformers import Qwen2VLForConditionalGeneration
            model_class = Qwen2VLForConditionalGeneration

        self.core_model = model_class.from_pretrained(
            model_path,
            load_in_8bit=load_in_8bit,
            torch_dtype="auto",
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
        self.core_model.eval()  # Set the model in eval mode explicitly even though not needed

        self.processor = AutoProcessor.from_pretrained(model_path)

    def support_image(self):
        return True

    def support_video(self):
        return True

    def chatbot_to_messages(self, chatbot) -> list[str]:
        messages = []
        content = []
        for user_msg, model_msg in chatbot:
            if isinstance(user_msg, (tuple, list)):  # query is media path
                if BaseModel.is_video_file(user_msg[0]):
                    content.append({"type": "video", "video": user_msg[0]})
                elif BaseModel.is_image_file(user_msg[0]) or BaseModel.is_base64_image(user_msg[0]):
                    content.append({"type": "image", "image": user_msg[0]})                
                else:
                    raise ValueError(f"Unsupported file type: {user_msg[0]}")
            elif isinstance(user_msg, str):  # query is text
                if user_msg:
                    content.append({"type": "text", "text": user_msg})
                    messages.append({"role": "user", "content": content})
                    content = []
                if model_msg:
                    messages.append({'role': 'assistant', 'content': model_msg})

        return messages

    def predict_simple_nostream(self, query: str, system: str, gen_kwargs: dict) -> str:
        image_url = gen_kwargs.get("image_url", None)
        dummy_chatbot = [
            [(image_url, ""), ""],
            [query, ""]]
        messages = self.chatbot_to_messages(dummy_chatbot)
        # Preprocess the inputs
        text_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(text=[text_prompt],
                                images=image_inputs,
                                videos=video_inputs,
                                padding=True,
                                return_tensors="pt")
        inputs = inputs.to("cuda")
        generated_ids = self.core_model.generate(**inputs, max_new_tokens=1024)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return output_text[0]

    def predict(self, chatbot: list[list[str | tuple]], params: dict):
        from time import time

        if len(chatbot) == 0 or not chatbot[-1][0] or chatbot[-1][1]:  # Empty user input or non-empty reply
            yield chatbot
        else:
            messages = self.chatbot_to_messages(chatbot)

            # Preprocess the inputs
            text_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            # Excepted output: '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe this image.<|im_end|>\n<|im_start|>assistant\n'

            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(text=[text_prompt],
                                    images=image_inputs,
                                    videos=video_inputs,
                                    padding=True,
                                    return_tensors="pt")
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
