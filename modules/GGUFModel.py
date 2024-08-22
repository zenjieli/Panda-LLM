import os
import os.path as osp
from typing import List
from llama_cpp_cuda_tensorcores import Llama, llama_chat_format
from modules.BaseModel import BaseModel
from utils.cjk_postprocessing import CJKPostprocessing


class GGUFModel(BaseModel):
    _stop_words = ['<|endoftext|>', '<|im_start|>', '<|im_end|>', '<|im_sep|>', '<Br>', '<br>']
    _chat_completion_params = ['temperature', 'top_p', 'top_k', 'repeat_penalty']

    def __init__(self, model_path, gpu_layers, n_ctx) -> None:
        super().__init__()

        # if model_path is a directory, find the first gguf file in the directory
        if osp.isfile(model_path):
            if model_path.lower().endswith(".gguf"):
                model_path = model_path
        elif osp.isdir(model_path):
            model_path = next((osp.join(model_path, f) for f in os.listdir(model_path) if f.lower().endswith(".gguf")), None)

        if model_path is None:
            raise FileNotFoundError(f".gguf file not found in {model_path}.")

        self.core_model = Llama(model_path=model_path, n_gpu_layers=gpu_layers, n_ctx=n_ctx)
        self._chat_formatter = self._find_chat_formatter()
        self._ctx_size = n_ctx

    def _find_chat_formatter(self) -> llama_chat_format.ChatFormatter:
        llm = self.core_model
        if self.core_model.chat_format == 'chatml':
            return llama_chat_format.format_chatml
        elif self.core_model.chat_format == 'mistral-instruct':
            return llama_chat_format.format_mistral_instruct
        else:
            template = llm.metadata["tokenizer.chat_template"]
            try:
                eos_token_id = int(llm.metadata["tokenizer.ggml.eos_token_id"])
            except:
                eos_token_id = llm.token_eos()
            try:
                bos_token_id = int(llm.metadata["tokenizer.ggml.bos_token_id"])
            except:
                bos_token_id = llm.token_bos()

            eos_token = llm._model.token_get_text(eos_token_id)
            bos_token = llm._model.token_get_text(bos_token_id)

            return llama_chat_format.Jinja2ChatFormatter(template=template, eos_token=eos_token, bos_token=bos_token)

    def check_token_count(self, token_count: int) -> bool:
        return token_count < self._ctx_size

    def try_tokenize(self, chatbot, system_prompt) -> List:
        tokenizer = self.core_model.tokenizer()
        if not tokenizer:
            return []

        if not self._chat_formatter:
            return []

        messages = BaseModel.chatbot_to_messages(chatbot, system_prompt)
        if not messages:
            return []

        return tokenizer.encode(self._chat_formatter(messages=messages).prompt)

    def predict(self, chatbot, params):
        if len(chatbot) == 0 or not chatbot[-1][0] or chatbot[-1][1]:  # Empty user input or non-empty reply
            yield chatbot
        else:
            model_params, system_prompt, enable_postprocessing = BaseModel.gather_params(params, self._chat_completion_params)

            messages = BaseModel.chatbot_to_messages(chatbot, system_prompt = system_prompt)
            output = self.core_model.create_chat_completion(messages=messages, stream=True, **model_params)

            postprocessor = CJKPostprocessing(enable_postprocessing)
            for item in output:
                if self.stop_event.is_set():
                    break

                new_token = item['choices'][0]['delta'].get('content')
                if new_token in self._stop_words:
                    break

                if new_token:
                    chatbot[-1][-1] += postprocessor.run(new_token)
                    yield chatbot

        self.stop_event.clear()
