"""
This example shows how to use vLLM for running offline inference with
the correct prompt format on vision language models for text generation.

For most models, the prompt format should follow corresponding examples
on HuggingFace model repository.
"""
from transformers import AutoTokenizer
import torch

from vllm import LLM, SamplingParams
from vllm.assets.image import ImageAsset
from vllm.assets.video import VideoAsset
from vllm.utils import FlexibleArgumentParser

# NOTE: The default `max_num_seqs` and `max_model_len` may result in OOM on
# lower-end GPUs.
# Unless specified, these settings have been tested to work on a single L4.

# Phi-3-Vision


def run_phi3v(question: str, modality: str):
    assert modality == "image"

    prompt = f"<|user|>\n<|image_1|>\n{question}<|end|>\n<|assistant|>\n"  # noqa: E501
    # Note: The default setting of max_num_seqs (256) and
    # max_model_len (128k) for this model may cause OOM.
    # You may lower either to run this example on lower-end GPUs.

    # In this example, we override max_num_seqs to 5 while
    # keeping the original context length of 128k.

    # num_crops is an override kwarg to the multimodal image processor;
    # For some models, e.g., Phi-3.5-vision-instruct, it is recommended
    # to use 16 for single frame scenarios, and 4 for multi-frame.
    #
    # Generally speaking, a larger value for num_crops results in more
    # tokens per image instance, because it may scale the image more in
    # the image preprocessing. Some references in the model docs and the
    # formula for image tokens after the preprocessing
    # transform can be found below.
    #
    # https://huggingface.co/microsoft/Phi-3.5-vision-instruct#loading-the-model-locally
    # https://huggingface.co/microsoft/Phi-3.5-vision-instruct/blob/main/processing_phi3_v.py#L194
    llm = LLM(
        model="microsoft/Phi-3.5-vision-instruct",
        trust_remote_code=True,
        dtype = torch.bfloat16,
        # max_model_len=4096,
        # max_num_seqs=2,
        # Note - mm_processor_kwargs can also be passed to generate/chat calls
        mm_processor_kwargs={"num_crops": 16},
    )
    stop_token_ids = None
    return llm, prompt, stop_token_ids

# InternVL
def run_internvl(question: str, modality: str):
    assert modality == "image"

    model_name = "OpenGVLab/InternVL2-2B"

    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        max_model_len=4096,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              trust_remote_code=True)
    messages = [{'role': 'user', 'content': f"<image>\n{question}"}]
    prompt = tokenizer.apply_chat_template(messages,
                                           tokenize=False,
                                           add_generation_prompt=True)

    # Stop tokens for InternVL
    # models variants may have different stop tokens
    # please refer to the model card for the correct "stop words":
    # https://huggingface.co/OpenGVLab/InternVL2-2B#service
    stop_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|end|>"]
    stop_token_ids = [tokenizer.convert_tokens_to_ids(i) for i in stop_tokens]
    return llm, prompt, stop_token_ids

# Qwen2-VL
def run_qwen2_vl(question: str, modality: str):
    assert modality == "image"

    model_name = "Qwen/Qwen2-VL-2B-Instruct"

    llm = LLM(
        model=model_name,
        enable_prefix_caching=True
        # max_model_len=4096,
        # max_num_seqs=5,
        # Note - mm_processor_kwargs can also be passed to generate/chat calls
        # mm_processor_kwargs={
        #     "min_pixels": 28 * 28,
        #     "max_pixels": 1280 * 28 * 28,
        # },
    )

    prompt = ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
              "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
              f"{question}<|im_end|>\n"
              "<|im_start|>assistant\n")
    stop_token_ids = None
    return llm, prompt, stop_token_ids


model_example_map = {
    "phi3_v": run_phi3v,
    "internvl_chat": run_internvl,
    "qwen2_vl": run_qwen2_vl
}


def get_multi_modal_input(args):
    """
    return {
        "data": image or video,
        "question": question,
    }
    """
    if args.modality == "image":
        # Input image and question
        image = ImageAsset("bird") \
            .pil_image.convert("RGB")
        img_question = "How many birds are there? Answer with a single number."

        return {
            "data": image,
            "question": img_question,
        }

    msg = f"Modality {args.modality} is not supported."
    raise ValueError(msg)


def main(args):
    model = args.model_type
    if model not in model_example_map:
        raise ValueError(f"Model type {model} is not supported.")

    modality = args.modality
    mm_input = get_multi_modal_input(args)
    data = mm_input["data"]
    question = mm_input["question"]

    llm, prompt, stop_token_ids = model_example_map[model](question, modality)

    # We set temperature to 0.2 so that outputs can be different
    # even when all prompts are identical when running batch inference.
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=1,
        stop_token_ids=stop_token_ids)

    assert args.num_prompts > 0
    if args.num_prompts == 1:
        # Single inference
        inputs = {
            "prompt": prompt,
            "multi_modal_data": {
                modality: data
            },
        }

    else:
        # Batch inference
        inputs = [{
            "prompt": prompt,
            "multi_modal_data": {
                modality: data
            },
        } for _ in range(args.num_prompts)]

    from time import time
    t0 = time()
    outputs = llm.generate(inputs, sampling_params=sampling_params)
    print(f"\nTime: {time() - t0}\n")

    t0 = time()
    outputs = llm.generate(inputs, sampling_params=sampling_params)
    print(f"\nTime: {time() - t0}\n")

    t0 = time()
    outputs = llm.generate(inputs, sampling_params=sampling_params)
    print(f"\nTime: {time() - t0}\n")

    for o in outputs:
        generated_text = o.outputs[0].text
        print(f"\n{generated_text}\n")


if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description='Demo on using vLLM for offline inference with '
        'vision language models for text generation')
    parser.add_argument('--model-type',
                        '-m',
                        type=str,
                        default="qwen2_vl",
                        choices=model_example_map.keys(),
                        help='Huggingface "model_type".')
    parser.add_argument('--num-prompts',
                        type=int,
                        default=1,
                        help='Number of prompts to run.')
    parser.add_argument('--modality',
                        type=str,
                        default="image",
                        choices=['image', 'video'],
                        help='Modality of the input.')
    parser.add_argument('--num-frames',
                        type=int,
                        default=16,
                        help='Number of frames to extract from the video.')
    args = parser.parse_args()
    main(args)

    import torch.distributed as dist
    if dist.is_initialized():
        dist.destroy_process_group()

