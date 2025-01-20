import os
import os.path as osp
from typing import List
from modules.base_model import BaseModel
from utils.cjk_postprocessing import CJKPostprocessing
from modules.model_factory import ModelFactory


@ModelFactory.register(".gguf$")
class GGUFModel(BaseModel):
    _stop_words = ['<|endoftext|>', '<|im_start|>', '<|im_end|>', '<|im_sep|>', '<Br>', '<br>']
    _chat_completion_params = ['temperature', 'top_p', 'top_k', 'repeat_penalty']

    def __init__(self, dir_or_filename, gpu_layers, n_ctx, **kwargs) -> None:
        from llama_cpp_cuda_tensorcores import Llama
        from huggingface_hub.constants import HF_HUB_CACHE

        super().__init__()

        # If model_path is a directory, find the first gguf file in the directory
        model_path = osp.join(HF_HUB_CACHE, dir_or_filename)
        if osp.isdir(model_path):
            # List all GGUF files in the directory
            files = [f for f in os.listdir(model_path) if os.path.isfile(
                os.path.join(model_path, f)) and f.lower().endswith(".gguf")]

            # Return the first file if there are any files in the directory
            if len(files) > 0:
                files.sort()
                model_path = os.path.join(model_path, files[0])
            else:
                model_path = None

        if model_path is None or not osp.exists(model_path):
            raise FileNotFoundError(f".gguf file not found in {model_path}.")

        self.core_model = Llama(model_path=model_path, n_gpu_layers=gpu_layers, n_ctx=n_ctx)
        self._chat_formatter = self._find_chat_formatter()
        self._ctx_size = n_ctx

    def _find_chat_formatter(self):
        from llama_cpp_cuda_tensorcores import llama_chat_format
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

    def get_meta_info(self) -> dict:
        from utils.model_utils import get_block_count_from_llama_meta

        block_count = get_block_count_from_llama_meta(self.core_model.metadata)
        return f'block count {block_count}' if block_count > 0 else ''

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
            model_params, system_prompt, enable_postprocessing = BaseModel.gather_params(
                params, self._chat_completion_params)

            messages = BaseModel.chatbot_to_messages(chatbot, system_prompt=system_prompt)
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

    def predict_simple_nostream(self, query: str, system: str, gen_kwargs: dict) -> str:
        """Simply predict without streaming. Useful for OpenAI API server.
        """
        messages = [{"role": "system", "content": system}] if system else []
        messages.append({"role": "user", "content": query})

        # remove key stopping_criteria from gen_kwargs
        gen_kwargs = {k: v for k, v in gen_kwargs.items() if k != "stopping_criteria"}
        output = self.core_model.create_chat_completion(messages=messages, stream=False, **gen_kwargs)
        return output["choices"][0]["message"].get("content")

    @classmethod
    def description(cls) -> str:
        return "GGUF"
