"""
Support LLaVA GGUF
For grounding, prompt "Please provide the bounding box coordinate of the region this sentence describe: ..."
"""

import os
import os.path as osp
from PIL import Image
from typing import List
from llama_cpp_cuda_tensorcores import Llama, llama_chat_format
from modules.BaseModel import BaseModel
import base64
from typing import Tuple
from llama_cpp_cuda_tensorcores import Llama, llama_chat_format


class LLaVAModel(BaseModel):
    _chat_completion_params = ['temperature', 'top_p', 'top_k']

    def __init__(self, model_path, gpu_layers, n_ctx) -> None:
        super().__init__()

        mmproj_filepath, model_filepath, chat_handler_class = self.parse_model_path(model_path)
        chat_handler = chat_handler_class(clip_model_path=mmproj_filepath)
        self.llm = Llama(
            model_path=model_filepath,
            chat_handler=chat_handler,
            n_ctx=n_ctx,
            n_gpu_layers=gpu_layers
        )

    def support_image(self):
        return True

    def parse_model_path(self, model_dir) -> Tuple[str, str, type[llama_chat_format.Llava15ChatHandler]]:
        mmproj_filepath = next((osp.join(model_dir, f) for f in os.listdir(model_dir) if f.lower().endswith(".gguf") and "mmproj" in f),
                               None)
        model_filepath = next((osp.join(model_dir, f) for f in os.listdir(model_dir) if f.lower().endswith(".gguf") and "mmproj" not in f),
                              None)
        if mmproj_filepath is not None and model_filepath is not None:
            chat_handler_class = llama_chat_format.NanoLlavaChatHandler if 'v1.6-34b' in osp.split(model_filepath)[1] \
                else llama_chat_format.Llava15ChatHandler
            return mmproj_filepath, model_filepath, chat_handler_class

        # If either file is not found
        raise FileNotFoundError("Required .gguf files not found in the model directory.")

    def try_draw_bbox_on_latest_picture(self, text) -> str | None:
        """
        Try to draw a bounding box on the latest image. Returns the new image file path if successful; otherwise, return None.
        """

        from modules.vlm_drawing import extract_bounding_box, draw_bounding_box
        import tempfile

        bbox = extract_bounding_box(text)
        if bbox is not None:
            # load self._latest_img_filepath to an image
            img = Image.open(self._latest_img_filepath)

            if img is not None:
                img = draw_bounding_box(img, bbox)
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
                    # Save the image to the temporary file
                    img.save(tmp_file.name)
                    return tmp_file.name

        return None

    def image_to_base64_data_uri(self, file_path):
        self._latest_img_filepath = file_path
        with open(file_path, "rb") as img_file:
            base64_data = base64.b64encode(img_file.read()).decode("utf-8")
            return f"data:image/png;base64,{base64_data}"

    def chatbot_to_messages(self, chatbot, system_prompt) -> List[str]:
        messages = [{"role": "system", "content": system_prompt}] if system_prompt else []
        for idx, (user_msg, model_msg) in enumerate(chatbot):
            if isinstance(user_msg, (tuple, list)):  # query is image path
                data_uri = self.image_to_base64_data_uri(user_msg[0])
                messages.append({"role": "user", "content": [{"type": "image_url", "image_url": {"url": data_uri}}]})
            else:  # query is text
                messages.append({"role": "user", "content": [{"type": "text", "text": user_msg}]})

            if model_msg and idx < len(chatbot) - 1:
                messages.append({'role': 'assistant', 'content': model_msg})

        return messages

    def predict(self, chatbot, params):
        if len(chatbot) == 0 or not chatbot[-1][0] or chatbot[-1][1]:  # Empty user input or non-empty reply
            yield chatbot
        else:
            _, system_prompt, _ = BaseModel.gather_params(params, self._chat_completion_params)

            response = self.llm.create_chat_completion(
                messages=LLaVAModel.chatbot_to_messages(self, chatbot, system_prompt=system_prompt),
                stream=True)

            for item in response:
                if self.stop_event.is_set():
                    break

                new_token = item["choices"][0]["delta"].get("content")
                if new_token:
                    chatbot[-1][-1] += new_token
                    yield chatbot

        self.stop_event.clear()

        saved_img_path = self.try_draw_bbox_on_latest_picture(chatbot[-1][-1])
        if saved_img_path is not None:
            chatbot.append((None, (saved_img_path,)))

        yield chatbot
