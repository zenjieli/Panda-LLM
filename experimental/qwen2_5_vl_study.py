from PIL import Image

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import os.path as osp

def generate_inputs_embeds(model, input_ids, image_grid_thw, pixel_values, image_embeds=None, attention_mask=None):
    inputs_embeds = model.model.embed_tokens(input_ids)
    if pixel_values is not None:
        pixel_values = pixel_values.type(model.visual.dtype)
        if image_embeds is None:
            image_embeds = model.visual(pixel_values, grid_thw=image_grid_thw)
        n_image_tokens = (input_ids == model.config.image_token_id).sum().item()
        n_image_features = image_embeds.shape[0]
        if n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )
        image_mask = (
            (input_ids == model.config.image_token_id)
            .unsqueeze(-1)
            .expand_as(inputs_embeds)
            .to(inputs_embeds.device)
        )
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

    return inputs_embeds


def main(model_path: str, img_path: str):
    # Load the model in half-precision on the available device(s)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, torch_dtype="auto", device_map="auto")
    processor = AutoProcessor.from_pretrained(model_path)

    # Count the number of trainable parameters
    print("\n\nCount the number of trainable parameters:")
    accum = 0
    for name, p in model.named_parameters():
        accum += p.numel()
        print(f"{name}: {list(p.shape)} {p.numel()}\tAccumulated: {accum}")

    # Image
    image = Image.open(img_path)
    conversation = [{"role": "user", "content": [{"type": "image", },
                                                 {"type": "text", "text": "What is in the image? A. Traffic B. Text; C. Persons D. Fire. Answer with a single letter."}]}]

    pre_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    # Excepted output: '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|><|im_end|>\n<|im_start|>assistant\n'
    inputs = processor(text=[pre_prompt], images=[image], padding=True, return_tensors="pt").to("cuda")

    inputs_embeds = generate_inputs_embeds(model, **inputs)
    inputs["inputs_embeds"] = inputs_embeds

    # Inference: Generation of the output
    output_ids = model.generate(**inputs,
                                max_new_tokens=1,
                                do_sample=False,
                                top_k=None,
                                top_p=None,
                                temperature=None)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    output_text = processor.batch_decode(generated_ids)
    print(output_text[0])


if __name__ == "__main__":
    model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
    img_path = osp.expanduser("~/Pictures/broker.png")
    main(model_id, img_path)
