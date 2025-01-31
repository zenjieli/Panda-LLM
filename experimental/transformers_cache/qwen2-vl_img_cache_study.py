from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
import os.path as osp
from time import time


def generate_inputs_embeds(model, input_ids, image_grid_thw, pixel_values, attention_mask=None):
    inputs_embeds = model.model.embed_tokens(input_ids)
    if pixel_values is not None:
        pixel_values = pixel_values.type(model.visual.get_dtype())
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
        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

    return inputs_embeds


def main():
    # Load the model in half-precision on the available device(s)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct", torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

    # Image
    url = osp.expanduser("~/Pictures/CCTV/00424.jpg")
    image = Image.open(url)

    conversations = [[{"role": "user", "content": [{"type": "image", },
                                                   {"type": "text", "text": "Is there a person in the image? Answer with Yes or No."}]}],
                     [{"role": "user", "content": [{"type": "image", },
                                                   {"type": "text", "text": "Is there a dog in the image? Answer with Yes or No."}]}],
                     [{"role": "user", "content": [{"type": "image", },
                                                   {"type": "text", "text": "Is there a car in the image? Answer with Yes or No."}]}],
                     [{"role": "user", "content": [{"type": "image", },
                                                   {"type": "text", "text": "Is there a person lying on the floor? Answer with Yes or No."}]}]]

    for conversation in conversations:
        pre_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        # Excepted output: '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|><|im_end|>\n<|im_start|>assistant\n'
        pre_prompt_inputs = processor(text=[pre_prompt], images=[image], padding=True, return_tensors="pt").to("cuda")
        time0 = time()
        inputs_embeds = generate_inputs_embeds(model, **pre_prompt_inputs)
        time_embeds = time() - time0
        # Inference: Generation of the output
        time0 = time()
        outputs = model(**pre_prompt_inputs, inputs_embeds=inputs_embeds)
        time_decoder_time = time() - time0
        next_token_ids = outputs.logits[:, -1, :].argmax(-1)
        output_text = processor.batch_decode(next_token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        print(f"Output: {output_text[0]}\tEmbedding generation time: {round(time_embeds*1000)} msec\tDecoder time: {round(time_decoder_time*1000)} msec")


if __name__ == "__main__":
    main()
