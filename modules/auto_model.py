from typing import List
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer
from threading import Thread
import torch
from peft import PeftModel
from modules.base_model import BaseModel


class AutoModel(BaseModel):
    """For unquantized models, GPTQ and AWQ using transformers
    """
    _chat_completion_params = ['temperature', 'top_p', 'top_k', 'repetition_penalty']
    def __init__(self, model_path, lora_path=None,load_in_8bit=False, **kwargs):
        super().__init__()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._tokenizer = AutoTokenizer.from_pretrained(model_path)

        # model.dtype will be set to torch.float16 if GPTQ is used
        self.core_model = AutoModelForCausalLM.from_pretrained(model_path, load_in_8bit=load_in_8bit, device_map='auto', torch_dtype="auto")
        if lora_path:
            self.core_model = PeftModel.from_pretrained(self.core_model, model_id=lora_path)
        self.core_model.eval()

        # If model_max_length is set in tokenizer, use it; otherwise, use 4*1024
        self._model_max_length = self._tokenizer.model_max_length if self._tokenizer.model_max_length < 400*1024 else 4*1024

    class StopOnTokens(StoppingCriteria):
        def __call__(
            self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
        ) -> bool:
            stop_ids = (
                [2, 6, 7, 8],
            )  # "<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|im_sep|>"
            for stop_id in stop_ids:
                if input_ids[0][-1] == stop_id:
                    return True
            return False

    def _tokenize(self, chatbot, system_prompt) -> torch.Tensor:
        messages = BaseModel.chatbot_to_messages(chatbot, system_prompt)
        return self._tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors='pt')

    def try_tokenize(self, chatbot, system_prompt) -> List:
        return self._tokenize(chatbot, system_prompt).squeeze().tolist()

    def predict(self, chatbot, params):
        if len(chatbot) == 0 or not chatbot[-1][0] or chatbot[-1][1]:  # Empty user input or non-empty reply
            yield chatbot
        else:
            model_params, system_prompt, _ = BaseModel.gather_params(params, self._chat_completion_params)
            stop = self.StopOnTokens()
            model_inputs = self._tokenize(chatbot, system_prompt).to(self.device)
            streamer = TextIteratorStreamer(
                self._tokenizer, timeout=60, skip_prompt=True, skip_special_tokens=True
            )
            generate_kwargs = {
                'input_ids': model_inputs,
                'streamer': streamer,
                'do_sample': True if model_params.get('top_k', 40) > 1 else False, # do_sample=True only makes sense if top_k > 1
                'stopping_criteria': StoppingCriteriaList([stop]),
                'max_length': self._model_max_length,
                **model_params
            }
            t = Thread(target=self.core_model.generate, kwargs=generate_kwargs)
            t.start()

            for new_token in streamer:
                if self.stop_event.is_set():
                    break

                if new_token != '':
                    chatbot[-1][-1] += new_token
                    yield chatbot

            t.join()
            self.stop_event.clear()

    @classmethod
    def description(self) -> str:
        return "Huggingface models"