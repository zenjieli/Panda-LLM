from vlm_complexity_calculation.calflops import calculate_flops
import torch
from transformers import AutoModelForCausalLM
from vlm_complexity_calculation.utils import *
from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from deepseek_vl2.utils.io import load_pil_images

def count_flops_deepseek(model_name,
                    image,
                    prompt,
                    seq_len=128,
                    device = 'cuda:0',
                    max_new_tokens = 1):
    
    vl_chat_processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(model_name)
    tokenizer = vl_chat_processor.tokenizer

    vl_gpt: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(model_name,
                                                                            trust_remote_code=True,
                                                                            device_map='auto',
                                                                            #max_memory={0: "15GB", 1:"15GB", 2:"5GB"},
                                                                            torch_dtype=torch.bfloat16)
    
    conversation = [
        {
            "role": "<|User|>",
            "content": "<image>\n" + f"<|▁pad▁|>" * seq_len
        },
        {"role": "<|Assistant|>", "content": ""},
    ]

    prepare_inputs = vl_chat_processor(
        conversations=conversation,
        images=[image],
        force_batchify=True,
        system_prompt=""
    ).to(vl_gpt.device)

    generation_config = dict(
        input_ids=prepare_inputs.input_ids,
        attention_mask=prepare_inputs.attention_mask,
        images=prepare_inputs.images,
        images_seq_mask=prepare_inputs.images_seq_mask,
        images_spatial_crop=prepare_inputs.images_spatial_crop,
        labels=prepare_inputs.labels,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True
    )

    flops, _, _, _ = calculate_flops(model=vl_gpt,
                                      forward_mode='generate',
                                      kwargs=generation_config,
                                      output_precision=4,
                                      output_unit='T')

    return flops