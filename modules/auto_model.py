from typing import List
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer, \
    TextIteratorStreamer, AutoConfig
from threading import Thread
import torch
from peft import PeftModel
from modules.base_model import BaseModel
from utils.postprocessing import CJKPostprocessing, PostprocessingGroup, MathPostprocessing


class AutoModel(BaseModel):
    """For unquantized models using transformers
    """
    _chat_completion_params = ["temperature"]

    def __init__(self, model_path, lora_path=None, load_in_8bit=False, **kwargs):
        super().__init__()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._tokenizer = AutoTokenizer.from_pretrained(model_path)
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

        # Check position embedding scaling
        rope_yarn = kwargs.get("rope_yarn", False)
        if rope_yarn:
            if hasattr(config, "rope_scaling") and (
                config.rope_scaling == None or config.rope_scaling["rope_type"] != "yarn"):
                config.rope_scaling = {
                    "rope_type": "yarn",
                    "factor": 4.0,
                    "original_max_position_embeddings": 32768
                }
                print("Long context enabled.")

        dtype = config.torch_dtype if hasattr(config, "torch_dtype") else "auto"

        # model.dtype will be set to torch.float16 if GPTQ is used
        try:
            self.core_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                load_in_8bit=load_in_8bit,
                device_map="auto",
                torch_dtype=dtype,
                attn_implementation="flash_attention_2"
                if self.supports_flash_attention and dtype in [torch.bfloat16, torch.float16]
                else "sdpa",
                trust_remote_code=True,
                config=config)
        except:
            self.core_model = AutoModelForImageTextToText.from_pretrained(
                model_path, load_in_8bit=load_in_8bit, device_map="auto", torch_dtype="auto", trust_remote_code=True)

        if lora_path:
            self.core_model = PeftModel.from_pretrained(self.core_model, model_id=lora_path)
        self.core_model.eval()

        # If model_max_length is set in tokenizer, use it; otherwise, use 4*1024
        self._model_max_length = self._tokenizer.model_max_length if self._tokenizer.model_max_length < 400 * 1024 else 4 * 1024

    def _tokenize(self, chatbot, system_prompt) -> torch.Tensor:
        messages = BaseModel.chatbot_to_messages(chatbot, system_prompt)
        return self._tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")

    def try_tokenize(self, chatbot, system_prompt) -> List:
        return self._tokenize(chatbot, system_prompt).squeeze().tolist()

    def predict(self, chatbot, params):
        if len(chatbot) == 0 or not chatbot[-1][0] or chatbot[-1][1]:  # Empty user input or non-empty reply
            yield chatbot, ""
        else:
            model_params, system_prompt, enable_postprocessing = BaseModel.gather_params(
                params, self._chat_completion_params)

            model_inputs = self._tokenize(chatbot, system_prompt).to(self.device)
            streamer = TextIteratorStreamer(
                self._tokenizer, timeout=60, skip_prompt=True, skip_special_tokens=True
            )
            generate_kwargs = {
                "input_ids": model_inputs,
                "streamer": streamer,
                "do_sample": True if model_params.get("temperature", 0.8) > 0 else False,
                "max_length": self._model_max_length,
                **model_params
            }
            t = Thread(target=self.core_model.generate, kwargs=generate_kwargs)
            t.start()

            postprocessors = PostprocessingGroup(CJKPostprocessing(enable_postprocessing), MathPostprocessing())
            for new_token in streamer:
                if self.stop_event.is_set():
                    break

                if new_token != "":
                    chatbot[-1][-1] += postprocessors(new_token)
                    yield chatbot, ""

            t.join()
            self.stop_event.clear()

    def predict_simple_nostream(self, query: str, system: str, gen_kwargs: dict) -> str:
        """Simply predict without streaming. Useful for OpenAI API server.
        """
        messages = [{"role": "system", "content": system}] if system else []
        messages.append({"role": "user", "content": query})
        model_inputs = self._tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt").to("cuda")

        gen_kwargs = {**gen_kwargs,
                      "input_ids": model_inputs,
                      "do_sample": True if gen_kwargs.get("temperature", 0.8) > 0 else False,
                      "max_length": self._model_max_length, }

        outputs = self.core_model.generate(**gen_kwargs)
        reply = self._tokenizer.decode(outputs[0][len(model_inputs[0]):], skip_special_tokens=True)

        return reply

    @classmethod
    def description(self) -> str:
        return "Huggingface models"
