from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer
from threading import Thread, Event
import torch
from modules.BaseModel import BaseModel

class GPTQModel(BaseModel):
    def __init__(self, model_path):
        super().__init__()

        self.__tokenizer = AutoTokenizer.from_pretrained(model_path)
        # model.dtype will be set to torch.float16 if GPTQ is used
        self.core_model = AutoModelForCausalLM.from_pretrained(model_path, device_map='cuda').eval()

        # If model_max_length is set in tokenizer, use it; otherwise, use 4*1024
        self.__model_max_length = self.__tokenizer.model_max_length if self.__tokenizer.model_max_length < 400*1024 else 4*1024

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

    def predict(self, history, top_p, temperature):
        if len(history) == 0 or not history[-1][0] or history[-1][1]:  # Empty user input or non-empty reply
            yield history
        else:
            stop = self.StopOnTokens()
            messages = []
            for idx, (user_msg, model_msg) in enumerate(history):
                if idx == len(history) - 1 and not model_msg:
                    messages.append({'role': 'user', 'content': user_msg})
                    break
                if user_msg:
                    messages.append({'role': 'user', 'content': user_msg})
                if model_msg:
                    messages.append({'role': 'assistant', 'content': model_msg})

            model_inputs = self.__tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True, return_tensors='pt'
            ).to(next(self.core_model.parameters()).device)
            streamer = TextIteratorStreamer(
                self.__tokenizer, timeout=60, skip_prompt=True, skip_special_tokens=True
            )
            generate_kwargs = {
                'input_ids': model_inputs,
                'streamer': streamer,
                'do_sample': True,
                'top_p': top_p,
                'temperature': temperature,
                'stopping_criteria': StoppingCriteriaList([stop]),
                'repetition_penalty': 1.2,
                'max_length': self.__model_max_length
            }
            t = Thread(target=self.core_model.generate, kwargs=generate_kwargs)
            t.start()

            for new_token in streamer:
                if self.stop_event.is_set():
                    break

                if new_token != '':
                    history[-1][-1] += new_token
                    yield history

            t.join()
            self.stop_event.clear()