import os.path as osp
from transformers import AutoProcessor
import torch.distributed as dist
from vllm import LLM
from vllm import SamplingParams
from qwen_vl_utils import process_vision_info
from time import time

MODEL_PATH = "Qwen/Qwen2-VL-2B-Instruct"

llm = LLM(
    model=MODEL_PATH,
    limit_mm_per_prompt={"image": 1, "video": 1},
    enable_prefix_caching=True
)

sampling_params = SamplingParams(
    top_k=1,
    max_tokens=100,
    stop_token_ids=[],
)

system_prompt_dict = {"role": "system", "content": "You are a helpful assistant."}
img_prompt_dict = {"type": "image",
                   "image": osp.expanduser("~/Pictures/CCTV/RidingBike-marc-kleen-WeBASN7ESOY-unsplash.jpg")}

all_messages = [[system_prompt_dict,
                 {
                     "role": "user",
                     "content": [
                         img_prompt_dict,
                         {"type": "text", "text": "Is there a tiger in the image? Answer with Yes or No."},
                     ],
                 }],
                [system_prompt_dict,
                 {
                     "role": "user",
                     "content": [
                         img_prompt_dict,
                         {"type": "text", "text": "Is there a bike in the image? Answer with Yes or No."},
                     ],
                 }],
                [system_prompt_dict,
                 {
                     "role": "user",
                     "content": [
                         img_prompt_dict,
                         {"type": "text", "text": "How many people are there in the image. Answer with a single number."},
                     ],
                 }]]

processor = AutoProcessor.from_pretrained(MODEL_PATH)

for messages in all_messages:
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    mm_data = {"image": image_inputs}

    llm_inputs = {"prompt": prompt, "multi_modal_data": mm_data}
    t0 = time()
    outputs = llm.generate([llm_inputs], sampling_params=sampling_params)
    print(f"\nTime: {time() - t0}\n")
    generated_text = outputs[0].outputs[0].text
    print("\n" + generated_text + "\n")

if dist.is_initialized():
    dist.destroy_process_group()