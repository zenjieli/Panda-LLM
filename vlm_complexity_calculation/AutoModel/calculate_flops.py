import torch
from PIL import Image
from vlm_complexity_calculation.calflops import calculate_flops
from vlm_complexity_calculation.utils import *
from transformers import AutoProcessor, AutoModelForCausalLM, AutoModelForImageTextToText

def count_flops_automodel(model_name,
                          image,
                          prompt,
                          seq_len=128,
                          device = 'cuda',
                          max_new_tokens = 1):

    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype="auto", trust_remote_code=True)
    except:
        model = AutoModelForImageTextToText.from_pretrained(model_name, device_map="auto", torch_dtype="auto", trust_remote_code=True)

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image"},
            ],
        },
    ]

    try:
        prompt_aux = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(images=[image], text=prompt_aux, return_tensors='pt').to(device, torch.float16)
    except:
        prompt_aux = processor.tokenizer.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(images=[image], texts=prompt_aux, return_tensors='pt').to(device, torch.float16)

    
    inputs["max_new_tokens"] = max_new_tokens

    if prompt == "":
        inputs = get_raw_input(processor.tokenizer, seq_len, inputs, device)

    flops, _, _, _ = calculate_flops(model=model,
                                      forward_mode = 'generate',
                                      kwargs = inputs,
                                      output_precision = 4,
                                      output_unit = 'T')
    
    return flops