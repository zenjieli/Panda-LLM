from .utils import *
import importlib
from vlm_complexity_calculation.calflops import calculate_flops
import torch
from vlm_complexity_calculation.utils import *
from transformers import AutoModel, AutoTokenizer

def split_model():
    device_map = dict()
    
    device_map['vision_model'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    
    for l in range(13):
        device_map[f'language_model.model.layers.{l}'] = 0
    
    for l in range(13, 31):
        device_map[f'language_model.model.layers.{l}'] = 1
    
    device_map[f'language_model.model.layers.31'] = 0
    device_map[f'language_model.model.norm'] = 0
    device_map['language_model.output'] = 0
    device_map['mlp1'] = 0

    return device_map


def count_flops_internvl2(model_name,
                          image,
                          prompt,
                          seq_len=128,
                          device = 'cuda',
                          max_new_tokens = 1,
                          num_slices = None):


    # If you want to load a model using multiple GPUs, please refer to the `Multiple GPUs` section.
    model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
        device_map='auto').eval()
    #model.to(device=device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)

    # set the max number of tiles in `max_num`
    if num_slices:
        pixel_values = load_image(image, max_num=num_slices).to(device=device, dtype=torch.bfloat16)
    else:
        pixel_values = load_image(image).to(device=device, dtype=torch.bfloat16)
    generation_config = dict(max_new_tokens=max_new_tokens, do_sample=False)

    # single-image single-round conversation (单图单轮对话)
    question = '<image>\n' + prompt

    libname = '.'.join(str(type(model)).split("'")[1].split('.')[:4] + ['conversation'])
    conversation = importlib.import_module(libname)

    num_patches_list = [pixel_values.shape[0]]

    img_context_token_id = tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')
    model.img_context_token_id = img_context_token_id

    template = conversation.get_conv_template(model.template)
    template.system_message = model.system_message
    eos_token_id = tokenizer.convert_tokens_to_ids(template.sep)

    template.append_message(template.roles[0], question)
    template.append_message(template.roles[1], None)
    query = template.get_prompt()

    for num_patches in num_patches_list:
        image_tokens = '<img>' + '<IMG_CONTEXT>' * model.num_image_token * num_patches + '</img>'
        query = query.replace('<image>', image_tokens, 1)

    model_inputs = tokenizer(query, return_tensors='pt')
    generation_config['input_ids'] = model_inputs['input_ids'].to(device=device)
    generation_config['eos_token_id'] = eos_token_id
    generation_config['attention_mask'] = model_inputs['attention_mask'].to(device=device)
    generation_config['pixel_values'] = pixel_values

    if prompt == "":
        generation_config = get_raw_input(tokenizer, seq_len, generation_config, device)

    flops, _, _, _ = calculate_flops(model=model,
                                      forward_mode = 'generate',
                                      kwargs = generation_config,
                                      output_precision = 4,
                                      output_unit = 'T')

    return flops