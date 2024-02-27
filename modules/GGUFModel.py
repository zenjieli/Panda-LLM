from llama_cpp_cuda_tensorcores import Llama
from modules.BaseModel import BaseModel


class GGUFModel(BaseModel):
    def __init__(self, model_path, gpu_layers) -> None:
        super().__init__()
        self.llm = Llama(model_path=model_path, chat_format="chatml",
                         n_gpu_layers=gpu_layers, n_ctx=8*1024)

    def predict(self, history, top_p, temperature):
        if len(history) == 0 or not history[-1][0] or history[-1][1]:  # Empty user input or non-empty reply
            yield history
        else:
            messages = []
            for idx, (user_msg, model_msg) in enumerate(history):
                if idx == len(history) - 1 and not model_msg:
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
                history[-1][-1] += new_token
                yield history

        self.stop_event.clear()
