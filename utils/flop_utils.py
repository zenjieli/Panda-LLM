from PIL import Image

def get_num_flops(model_name):
    image = Image.new('RGB', (1920, 1080))
    seq_len = 128
    query = ""
    max_new_tokens = 1

    result = ""

    if "openbmb" in model_name.lower():
        from vlm_complexity_calculation.MiniCPM_V.calculate_flops import count_flops_minicpm
        result = count_flops_minicpm(model_name=model_name,
                                     image=image,
                                     prompt=query,
                                     seq_len=seq_len,
                                     max_new_tokens=max_new_tokens
                                     )

    elif "phi" in model_name.lower():
        from vlm_complexity_calculation.Phi_Vision.calculate_flops import count_flops_phi
        result = count_flops_phi(model_name=model_name,
                                 image=image,
                                 prompt=query,
                                 seq_len=seq_len,
                                 max_new_tokens=max_new_tokens
                                 )
    
    elif "internvl2" in model_name.lower():
        from vlm_complexity_calculation.InternVL2.calculate_flops import count_flops_internvl2
        result = count_flops_internvl2(model_name=model_name,
                                       image=image,
                                       prompt=query,
                                       seq_len=seq_len,
                                       max_new_tokens=max_new_tokens
                                       )

    elif "llava-hf" in model_name:
        from vlm_complexity_calculation.LlavaNext.calculate_flops import count_flops_llavanext
        result = count_flops_llavanext(model_name=model_name,
                                       image=image,
                                       prompt=query,
                                       seq_len=seq_len,
                                       max_new_tokens=max_new_tokens
                                       )

    elif "qwen2" in model_name.lower():
        from vlm_complexity_calculation.Qwen2.calculate_flops import count_flops_qwen2
        result = count_flops_qwen2(model_name=model_name,
                                  image=image,
                                  prompt=query,
                                  seq_len=seq_len,
                                  max_new_tokens=max_new_tokens
                                  )

    elif "ovis" in model_name.lower():
        from vlm_complexity_calculation.Ovis2.calculate_flops import count_flops_ovis2
        result = count_flops_ovis2(model_name=model_name,
                                       image=image,
                                       prompt=query,
                                       seq_len=seq_len,
                                       max_new_tokens=max_new_tokens
                                       )
    
    elif "deepseek" in model_name.lower():
        from vlm_complexity_calculation.Deepseek.calculate_flops import count_flops_deepseek
        result = count_flops_deepseek(model_name=model_name,
                                       image=image,
                                       prompt=query,
                                       seq_len=seq_len,
                                       max_new_tokens=max_new_tokens
                                       )
    
    else:
        from vlm_complexity_calculation.AutoModel.calculate_flops import count_flops_automodel
    try:
        result = count_flops_automodel(model_name=model_name,
                                    image=image,
                                    prompt=query,
                                    seq_len=seq_len,
                                    max_new_tokens=max_new_tokens
                                    )
    except:
            return "Calculation of FLOPs is not supported for this model"

    return f"FLOPs: {result}"