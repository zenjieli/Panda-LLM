from time import time
from PIL import Image
import torch
import os.path as osp
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor

# Load the model in half-precision on the available device(s)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct", torch_dtype=torch.bfloat16, device_map="cuda"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

# Image
url = osp.expanduser("~/Pictures/CCTV/RidingBike-marc-kleen-WeBASN7ESOY-unsplash.jpg")
image = Image.open(url)

conversations = [[{"role": "user",
                 "content": [{"type": "image"}, {"type": "text", "text": "Is there a tiger in the image? Answer with Yes or No."}]}],
                 [{"role": "user",
                 "content": [{"type": "image"}, {"type": "text", "text": "Is there a bike in the image? Answer with Yes or No."}]}],
                 [{"role": "user",
                 "content": [{"type": "image"}, {"type": "text", "text": "How many people are there in the image. Answer with a single number."}]}]]


for conversation in conversations:
    # Preprocess the inputs
    text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    # Excepted output: '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe this image.<|im_end|>\n<|im_start|>assistant\n'

    inputs = processor(
        text=[text_prompt], images=[image], padding=True, return_tensors="pt"
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    t0 = time()
    output_ids = model.generate(**inputs, max_new_tokens=128)
    print(f"\nTime: {time() - t0}\n")
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    print(processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0])
