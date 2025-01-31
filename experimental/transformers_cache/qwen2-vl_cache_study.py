import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, StaticCache
import os.path as osp


def reset_cache(prompt_cache: StaticCache, pre_prompt_len: int) -> None:
    for k_cache in prompt_cache.key_cache:
        k_cache[:, :, pre_prompt_len:, :] = 0

    for v_cache in prompt_cache.value_cache:
        v_cache[:, :, pre_prompt_len:, :] = 0


def main():
    # Load the model in half-precision on the available device(s)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct", torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

    # Image
    url = osp.expanduser("~/Pictures/bence-horvai-3-c_auIU2tQ-unsplash_small.jpg")
    image = Image.open(url)

    conversation = [{"role": "user", "content": [{"type": "image", },
                                                 {"type": "text", "text": ""}]}]
    # {"type": "text", "text": "Describe this image."}]}]

    pre_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    # Excepted output: '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|><|im_end|>\n<|im_start|>assistant\n'
    pre_prompt_inputs = processor(text=[pre_prompt], images=[image], padding=True, return_tensors="pt").to("cuda")
    prefix_len = len(pre_prompt_inputs.input_ids[0]) - 5
    prompt_cache = StaticCache(config=model.config, batch_size=1, max_cache_len=prefix_len +
                               256, device="cuda", dtype=model.config.torch_dtype)
    with torch.no_grad():
        prompt_cache = model(**pre_prompt_inputs, past_key_values=prompt_cache, use_cache=True).past_key_values

    conversation = [{"role": "user", "content": [{"type": "image", },
                                                 {"type": "text", "text": "Is this image a portrait?"}]}]
    text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(text=[text_prompt], images=[image], padding=True, return_tensors="pt").to("cuda")
    reset_cache(prompt_cache, prefix_len)

    # Inference: Generation of the output
    output_ids = model.generate(**inputs,
                                max_new_tokens=128,
                                past_key_values=prompt_cache,
                                do_sample=False,
                                top_k = None,
                                top_p = None,
                                temperature = None,
                                return_dict_in_generate=True).sequences
    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(inputs.input_ids, output_ids)
    ]
    output_text = processor.batch_decode(generated_ids)
    print(output_text)


if __name__ == "__main__":
    main()
