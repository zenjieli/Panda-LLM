from llama_cpp_cuda import Llama

llm = Llama(
    model_path="models/Yi-34B-Chat-GGUF/yi-34b-chat.Q4_K_M.gguf",
    chat_format="chatml",
    n_gpu_layers=30
)
output = llm.create_chat_completion(
    messages=[
        {"role": "system", "content": "You are an assistant who perfectly describes images."},
        {
            "role": "user",
            "content": "Who are you?"
        }
    ],
    stream=True,
    temperature=0.7,
)

for item in output:
    content = item['choices'][0]['delta'].get('content')

    if content:
        print(content, end='')
