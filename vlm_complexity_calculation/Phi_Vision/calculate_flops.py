from vlm_complexity_calculation.calflops import calculate_flops
import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image
from vlm_complexity_calculation.utils import *

def count_flops_phi(model_name,
                    image,
                    prompt,
                    seq_len = 128,
                    device = 'cuda',
                    max_new_tokens = 1,
                    num_slices = None):

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map='auto',
        trust_remote_code=True,
        torch_dtype="auto",
        _attn_implementation='flash_attention_2'
    )

    if num_slices:
        processor = AutoProcessor.from_pretrained(model_name,
                                                  trust_remote_code=True,
                                                  num_crops=num_slices
                                                  )
    else:
        processor = AutoProcessor.from_pretrained(model_name,
                                                  trust_remote_code=True,
                                                  )

    messages = [
        {"role": "user", "content": f"<|image_1|>\n" + prompt},
    ]

    prompt_aux = processor.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = processor(prompt_aux, [image]).to(device)
    inputs["max_new_tokens"] = max_new_tokens
    inputs["eos_token_id"] = processor.tokenizer.eos_token_id

    if prompt == "":
        inputs = get_raw_input(processor.tokenizer, seq_len, inputs, device)

    flops, _, _, _ = calculate_flops(model=model,
                                      forward_mode = 'generate',
                                      kwargs = inputs,
                                      output_precision = 4,
                                      output_unit = 'T')

    return flops