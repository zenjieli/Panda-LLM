from vlm_complexity_calculation.calflops import calculate_flops
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer, AutoProcessor
from vlm_complexity_calculation.utils import *

def get_inputs_minicpmv(image,
                        model,
                        tokenizer,
                        prompt,
                        seq_len=128,
                        max_new_tokens = 1):
    images = [image]
    content = (
            tokenizer.im_start
            + tokenizer.unk_token * model.config.query_num
            + tokenizer.im_end
            + "\n"
    )

    if prompt != "":
        content += prompt
    else:
        content += tokenizer.unk_token * seq_len

    final_input = "<用户>" + content + "<AI>"

    inputs = {"data_list": [final_input],
              "img_list": [images],
              "tokenizer": tokenizer,
              "max_new_tokens": max_new_tokens}

    return inputs

def get_inputs_minicpmv_2(image,
                          model,
                          tokenizer,
                          prompt,
                          seq_len = 128,
                          max_new_tokens = 1,
                          num_slices = None):

    if num_slices:
        model.config.max_slice_nums = num_slices

    if model.config.slice_mode:
        images, final_placeholder = model.get_slice_image_placeholder(
            image, tokenizer
        )

        if prompt != "":
            content = final_placeholder + "\n" + prompt
        else:
            content = final_placeholder + "\n" + tokenizer.unk_token * seq_len

        final_input = "<用户>" + content + "<AI>"

        inputs = {"data_list": [final_input],
                  "img_list": [images],
                  "tokenizer": tokenizer,
                  "max_new_tokens": max_new_tokens}

        return inputs

    else:
        return get_inputs_minicpmv(image, model, tokenizer, prompt, max_new_tokens)

def get_inputs_processor(image,
                         model,
                         prompt):

    msgs = [{'role': 'user', 'content': [image, prompt]}]
    processor = AutoProcessor.from_pretrained(model.config._name_or_path, trust_remote_code=True)

    msg = msgs[0]
    content = msg["content"]
    cur_msgs = []
    for c in content:
        if isinstance(c, Image.Image):
            cur_msgs.append("(<image>./</image>)")
        elif isinstance(c, str):
            cur_msgs.append(c)
    msg["content"] = "\n".join(cur_msgs)

    prompt_aux = processor.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    images = [image]

    return prompt_aux, images, processor

def get_inputs_minicpmv_2_6(image,
                            model,
                            tokenizer,
                            prompt,
                            seq_len = 128,
                            max_new_tokens = 1,
                            num_slices = None):

    prompt_aux, images, processor = get_inputs_processor(image, model, prompt)

    if num_slices:
        inputs = processor(
            [prompt_aux],
            [images],
            max_slice_nums=num_slices,
            return_tensors="pt"
        ).to(model.device)
    else:
        inputs = processor(
            [prompt_aux],
            [images],
            return_tensors="pt"
        ).to(model.device)

    inputs["tokenizer"] = tokenizer
    inputs["max_new_tokens"] = max_new_tokens

    if prompt == "":
        inputs = get_raw_input(processor.tokenizer, seq_len, inputs, model.device)

    if 'image_sizes' in inputs:
        del inputs['image_sizes']

    return inputs

def get_inputs_minicpmv_2_5_llama3(image,
                                   model,
                                   tokenizer,
                                   prompt,
                                   seq_len=128,
                                   max_new_tokens = 1,
                                   num_slices = None):

    if num_slices:
        model.config.slice_config.max_slice_nums = num_slices

    prompt_aux, images, processor = get_inputs_processor(image, model, prompt)

    inputs = processor(prompt_aux, images, return_tensors="pt").to(model.device)

    if prompt == "":
        inputs = get_raw_input(processor.tokenizer, seq_len, inputs, model.device)

    params = dict()
    params["model_inputs"] = inputs
    params["tokenizer"] = tokenizer
    params["max_new_tokens"] = max_new_tokens

    return params

def split_model():
    device_map=dict()
    device_map['vpm.embeddings'] = 0

    for i in range(0, 15):
        device_map[f'vpm.encoder.layers.{i}'] = 0
    
    for i in range(15, 28):
        device_map[f'vpm.encoder.layers.{i}'] = 2
    
    device_map['vpm.post_layernorm'] = 2
    device_map['resampler'] = 0
    device_map['llm.model.embed_tokens'] = 0
    device_map['llm.model.rotary_emb'] = 0
    device_map['llm.model.layers.0'] = 0

    for i in range(1, 31):
        device_map[f'llm.model.layers.{i}'] = 1
    
    device_map[f'llm.model.layers.31'] = 0
    device_map['llm.model.norm'] = 0
    device_map['llm.lm_head'] = 0

    return device_map


def count_flops_minicpm(model_name,
                        image,
                        prompt,
                        seq_len=128,
                        device = 'cuda',
                        max_new_tokens = 1,
                        num_slices = None):

    model = AutoModel.from_pretrained(model_name,
                                      trust_remote_code=True,
                                      torch_dtype=torch.bfloat16,
                                      attn_implementation='flash_attention_2',
                                      device_map='auto'
                                      )
    model = model.to(dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if model_name == "openbmb/MiniCPM-V":
        inputs = get_inputs_minicpmv(image, model, tokenizer, prompt, seq_len, max_new_tokens)

    elif model_name == "openbmb/MiniCPM-V-2":
        inputs = get_inputs_minicpmv_2(image, model, tokenizer, prompt, seq_len, max_new_tokens, num_slices)

    elif "2_6" in model_name:
        inputs = get_inputs_minicpmv_2_6(image, model, tokenizer, prompt, seq_len, max_new_tokens, num_slices)

    elif model_name == "openbmb/MiniCPM-Llama3-V-2_5":
        inputs = get_inputs_minicpmv_2_5_llama3(image, model, tokenizer, prompt, seq_len, max_new_tokens, num_slices)

    else:
        print("Model not recognized. Available models from MiniCPM are openbmb/MiniCPM-V, openbmb/MiniCPM-V-2, openbmb/MiniCPM-V-2_6 are openbmb/MiniCPM-Llama3-V-2_5")

    flops, _, _, _ = calculate_flops(model=model,
                                      forward_mode='generate',
                                      kwargs=inputs,
                                      output_precision=4,
                                      output_unit='T')

    return flops