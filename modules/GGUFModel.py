from llama_cpp_cuda_tensorcores import Llama
from modules.BaseModel import BaseModel


class GGUFModel(BaseModel):
    def __init__(self, model_path, gpu_layers) -> None:
        super().__init__()
        self.llm = Llama(model_path=model_path, chat_format="chatml",
                         n_gpu_layers=gpu_layers, n_ctx=8*1024)

    def predict(self, chatbot, task_history, top_p, temperature):
        if len(chatbot) == 0 or not chatbot[-1][0] or chatbot[-1][1]:  # Empty user input or non-empty reply
            yield chatbot
        else:
            messages = []
            for idx, (user_msg, model_msg) in enumerate(chatbot):
                if idx == len(chatbot) - 1 and not model_msg:
                    messages.append({'role': 'user', 'content': user_msg})
                    break
                if user_msg:
                    messages.append({'role': 'user', 'content': user_msg})
                if model_msg:
                    messages.append({'role': 'assistant', 'content': model_msg})

            output = self.llm.create_chat_completion(messages=messages, stream=True, top_p=top_p, temperature=temperature)

        for item in output:
            if self.stop_event.is_set():
                break

            new_token = item['choices'][0]['delta'].get('content')
            if new_token:
                chatbot[-1][-1] += new_token
                yield chatbot

        self.stop_event.clear()
