"""
Support InternVideo with transformers library
"""
import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
from modules.base_model import BaseModel
from modules.model_factory import ModelFactory


@ModelFactory.register("internvideo.*")
class InternVideoModel(BaseModel):
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(self, model_path, **kwargs) -> None:
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.core_model = AutoModel.from_pretrained(
            model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto")
        self.generation_config = dict(max_new_tokens=1024, do_sample=True)

    def support_image(self):
        return False

    def support_video(self):
        return True

    @staticmethod
    def build_transform(input_size):
        MEAN, STD = InternVideoModel.IMAGENET_MEAN, InternVideoModel.IMAGENET_STD
        transform = T.Compose([T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img), T.Resize(
            (input_size, input_size), interpolation=InterpolationMode.BICUBIC), T.ToTensor(), T.Normalize(mean=MEAN, std=STD)])
        return transform

    @staticmethod
    def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
        best_ratio_diff = float("inf")
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    @staticmethod
    def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        # calculate the existing image aspect ratio
        target_ratios = set((i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1)
                            for j in range(1, n + 1) if i * j <= max_num and i * j >= min_num)
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        # find the closest aspect ratio to the target
        target_aspect_ratio = InternVideoModel.find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size)

        # calculate the target width and height
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        # resize the image
        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = ((i % (target_width // image_size)) * image_size, (i // (target_width // image_size)) * image_size,
                   ((i % (target_width // image_size)) + 1) * image_size, ((i // (target_width // image_size)) + 1) * image_size)
            # split the image
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        assert len(processed_images) == blocks
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        return processed_images

    @staticmethod
    def load_image(image, input_size=448, max_num=6):
        transform = InternVideoModel.build_transform(input_size=input_size)
        images = InternVideoModel.dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        return pixel_values

    @staticmethod
    def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
        if bound:
            start, end = bound[0], bound[1]
        else:
            start, end = -100000, 100000
        start_idx = max(first_idx, round(start * fps))
        end_idx = min(round(end * fps), max_frame)
        seg_size = float(end_idx - start_idx) / num_segments
        frame_indices = np.array([int(start_idx + (seg_size / 2) + np.round(seg_size * idx)) for idx in range(num_segments)])
        return frame_indices

    @staticmethod
    def get_num_frames_by_duration(duration):
        local_num_frames = 4
        num_segments = int(duration // local_num_frames)
        if num_segments == 0:
            num_frames = local_num_frames
        else:
            num_frames = local_num_frames * num_segments

        num_frames = min(512, num_frames)
        num_frames = max(128, num_frames)

        return num_frames

    @staticmethod
    def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=32, get_frame_by_duration=False):
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        max_frame = len(vr) - 1
        fps = float(vr.get_avg_fps())

        pixel_values_list, num_patches_list = [], []
        transform = InternVideoModel.build_transform(input_size=input_size)
        if get_frame_by_duration:
            duration = max_frame / fps
            num_segments = InternVideoModel.get_num_frames_by_duration(duration)
        frame_indices = InternVideoModel.get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
        for frame_index in frame_indices:
            img = Image.fromarray(vr[frame_index].asnumpy()).convert("RGB")
            img = InternVideoModel.dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
            pixel_values = [transform(tile) for tile in img]
            pixel_values = torch.stack(pixel_values)
            num_patches_list.append(pixel_values.shape[0])
            pixel_values_list.append(pixel_values)
        pixel_values = torch.cat(pixel_values_list)
        return pixel_values, num_patches_list

    # evaluation setting
    max_num_frames = 512
    generation_config = dict(
        do_sample=False,
        temperature=0.0,
        max_new_tokens=1024,
        top_p=0.1,
        num_beams=1
    )

    def chatbot_to_messages(self, chatbot) -> list[str]:
        messages = ""
        pixel_values = None
        num_patches_list = None
        for user_msg, model_msg in chatbot:
            if isinstance(user_msg, (tuple, list)):  # query is media path
                if BaseModel.is_video_file(user_msg[0]):
                    pixel_values, num_patches_list = InternVideoModel.load_video(
                        user_msg[0], num_segments=128, max_num=1, get_frame_by_duration=True)
                    pixel_values = pixel_values.to(torch.bfloat16).to(self.core_model.device)
                    messages += "".join([f"Frame{i+1}: <image>\n" for i in range(len(num_patches_list))])
                elif BaseModel.is_image_file(user_msg[0]):
                    raise NotImplementedError("Image input is not supported yet.")
                else:
                    raise ValueError(f"Unsupported file type: {user_msg[0]}")
            elif isinstance(user_msg, str):  # query is text
                if user_msg:
                    messages += user_msg
                if model_msg:
                    messages += model_msg

        return messages, pixel_values, num_patches_list

    def predict(self, chatbot: list[list[str | tuple]], params: dict):
        from time import time

        if len(chatbot) == 0 or not chatbot[-1][0] or chatbot[-1][1]:  # Empty user input or non-empty reply
            yield chatbot
        else:
            messages, pixel_values, num_patches_list = self.chatbot_to_messages(chatbot)

            t0 = time()
            generation_config = dict(do_sample=False, temperature=0.0, max_new_tokens=1024, top_p=0.1, num_beams=1)
            output = self.core_model.chat(self.tokenizer, pixel_values, messages, generation_config,
                                          num_patches_list=num_patches_list, history=None, return_history=False)
            chatbot[-1][-1] = output
            token_count = len(output)
            summary = f"New tokens: {token_count}; Speed: {token_count / (time() - t0):.1f} tokens/sec"
            yield chatbot, summary
