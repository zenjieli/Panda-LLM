import torch
from PIL import Image
from vlm_complexity_calculation.calflops import calculate_flops
from vlm_complexity_calculation.utils import *

def count_flops_llavanext(model_name,
                          image,
                          prompt,
                          seq_len=128,
                          device = 'cuda',
                          max_new_tokens = 1):
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image"},
            ],
        },
    ]

    if "onevision" in model_name:
        from transformers import LlavaOnevisionForConditionalGeneration, AutoProcessor

        processor = AutoProcessor.from_pretrained(model_name)
        model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map='auto',
            attn_implementation='flash_attention_2'
        )

    else:
        from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

        processor = LlavaNextProcessor.from_pretrained(model_name)
        model = LlavaNextForConditionalGeneration.from_pretrained(model_name,
                                                              torch_dtype=torch.float16,
                                                              low_cpu_mem_usage=True,
                                                              device_map='cuda:0',
                                                              attn_implementation='flash_attention_2')

    #model.to(device)

    prompt_aux = processor.apply_chat_template(conversation, add_generation_prompt=True)

    inputs = processor(images=image, text=prompt_aux, return_tensors='pt').to(device, torch.float16)
    inputs["max_new_tokens"] = max_new_tokens

    if prompt == "":
        inputs = get_raw_input(processor.tokenizer, seq_len, inputs, device)

    flops, _, _, _ = calculate_flops(model=model,
                                      forward_mode = 'generate',
                                      kwargs = inputs,
                                      output_precision = 4,
                                      output_unit = 'T')
    
    return flops