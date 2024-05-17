import base64
import os.path as osp
from llama_cpp_cuda_tensorcores import Llama
from llama_cpp_cuda_tensorcores.llama_chat_format import Llava15ChatHandler

def image_to_base64_data_uri(file_path):
    with open(file_path, "rb") as img_file:
        base64_data = base64.b64encode(img_file.read()).decode("utf-8")
        return f"data:image/png;base64,{base64_data}"

chat_handler = Llava15ChatHandler(clip_model_path="models/hf/LLaVA-v1.6-vicuna-13b/mmproj-model-f16.gguf")
llm = Llama(
    model_path="models/hf/LLaVA-v1.6-vicuna-13b/llava-v1.6-vicuna-13b.Q5_K_M.gguf",
    chat_handler=chat_handler,
    # n_ctx=2048,  # n_ctx should be increased to accommodate the image embedding
    n_ctx=4096,
    n_gpu_layers=-1
)

data_uri = image_to_base64_data_uri(osp.expanduser("~/Pictures/traffic/Highway_billboard.jpg"))
response = llm.create_chat_completion(
    messages=[
        {"role": "system", "content": "You are an assistant who perfectly describes images."},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_uri}}
            ]
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"}
            ]
        }
    ],
    stream=True
)
# print(response["choices"][0]["message"]["content"])
for item in response:
    new_token = item["choices"][0]["delta"].get("content")
    if new_token in [["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|im_sep|>", "<Br>", "<br>"]]:
        break

    if new_token:
        print(new_token, end=" ")
