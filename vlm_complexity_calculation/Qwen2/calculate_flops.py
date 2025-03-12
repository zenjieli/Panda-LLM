from vlm_complexity_calculation.calflops import calculate_flops
import torch
from vlm_complexity_calculation.utils import *

def count_flops_qwen2(model_name,
                    image,
                    prompt,
                    seq_len=128,
                    device = 'cuda:0',
                    max_new_tokens = 1):

    if '2.5' in model_name:
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype="auto", device_map='auto', attn_implementation="flash_attention_2"
        )
    
    else:
        from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype="auto", device_map='auto', attn_implementation="flash_attention_2"
        )

    # default processer
    processor = AutoProcessor.from_pretrained(model_name)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image"
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, add_generation_prompt=True
    )
    inputs = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(device)
    inputs['max_new_tokens'] = max_new_tokens

    if prompt == "":
        inputs = get_raw_input(processor.tokenizer, seq_len, inputs, device)

    flops, _, _, _ = calculate_flops(model=model,
                                      forward_mode='generate',
                                      kwargs=inputs,
                                      output_precision=4,
                                      output_unit='T')

    return flops