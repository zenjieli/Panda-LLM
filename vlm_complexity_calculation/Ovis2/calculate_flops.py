from vlm_complexity_calculation.calflops import calculate_flops
import torch
from transformers import AutoModelForCausalLM
from vlm_complexity_calculation.utils import *

def count_flops_ovis2(model_name,
                    image,
                    prompt,
                    seq_len=128,
                    device = 'cuda:0',
                    max_new_tokens = 1):
    
    model = AutoModelForCausalLM.from_pretrained(model_name,
                                             torch_dtype=torch.bfloat16,
                                             multimodal_max_length=32768,
                                             trust_remote_code=True).cuda()
    
    text_tokenizer = model.get_text_tokenizer()
    visual_tokenizer = model.get_visual_tokenizer()
    max_partition = 9

    _, input_ids, pixel_values = model.preprocess_inputs(prompt, [image], max_partition=max_partition)
    attention_mask = torch.ne(input_ids, text_tokenizer.pad_token_id)
    input_ids = input_ids.unsqueeze(0).to(device=model.device)
    attention_mask = attention_mask.unsqueeze(0).to(device=model.device)
    if pixel_values is not None:
        pixel_values = pixel_values.to(dtype=visual_tokenizer.dtype, device=visual_tokenizer.device)
    pixel_values = [pixel_values]

    if prompt == "":
        inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}
        inputs = get_raw_input(text_tokenizer, seq_len, inputs, device)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        top_p=None,
        top_k=None,
        temperature=None,
        repetition_penalty=None,
        eos_token_id=model.generation_config.eos_token_id,
        pad_token_id=text_tokenizer.pad_token_id,
        use_cache=True,
        inputs=input_ids,
        pixel_values=pixel_values,
        attention_mask=attention_mask
    )   

    flops, _, _, _ = calculate_flops(model=model,
                                      forward_mode='generate',
                                      kwargs=gen_kwargs,
                                      output_precision=4,
                                      output_unit='T')

    return flops