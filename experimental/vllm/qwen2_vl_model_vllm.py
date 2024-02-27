"""
    Support Qwen2-VL Model with vLLM
"""
from PIL import Image
from vllm import LLM, SamplingParams
import logging
from time import time
import torch


class Qwen2VLModelVLLM:
    def __init__(self, model_path) -> None:
        self.core_model = LLM(model=model_path, dtype=torch.bfloat16)
        logger = logging.getLogger('vllm.inputs.preprocess')
        logger.setLevel(logging.WARNING)
        self.elaspsed_time = 0
        self.timing_count = 0

    def infer(self, text_input: str, img_filepath: str, vision_hidden_states=None) -> str:
        image = Image.open(img_filepath).convert("RGB")

        sampling_params = SamplingParams(temperature=0, max_tokens=20, stop_token_ids=None)

        prompt = ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                  "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
                  f"{text_input}<|im_end|>\n"
                  "<|im_start|>assistant\n")

        inputs = {"prompt": prompt, "multi_modal_data": {"image": image}}
        t0 = time()
        outputs = self.core_model.generate(inputs, sampling_params=sampling_params, use_tqdm=False)
        self.elaspsed_time += time() - t0
        self.timing_count += 1

        if self.timing_count == 500:
            print(f"Average inference time: {self.elaspsed_time / self.timing_count}")

        return outputs[0].outputs[0].text, None, 1
