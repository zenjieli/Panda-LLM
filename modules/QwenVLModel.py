from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer
import secrets
import os.path as osp
from pathlib import Path
import tempfile
from modules.BaseModel import BaseModel


class QwenVLModel(BaseModel):
    PUNCTUATION = "！？。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."

    def __init__(self, model_path):
        super().__init__()

        self.__tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        # self.__llm.dtype will be set to torch.float16 for GPTQ models
        self.__llm = AutoModelForCausalLM.from_pretrained(model_path, device_map='cuda', trust_remote_code=True).eval()

        self.__uploaded_file_dir = osp.join(tempfile.gettempdir(), "gradio")

    def support_image(self):
        return True

    def _parse_text(self, text):
        lines = text.split("\n")
        lines = [line for line in lines if line != ""]
        count = 0
        for i, line in enumerate(lines):
            if "```" in line:
                count += 1
                items = line.split("`")
                if count % 2 == 1:
                    lines[i] = f'<pre><code class="language-{items[-1]}">'
                else:
                    lines[i] = f"<br></code></pre>"
            else:
                if i > 0:
                    if count % 2 == 1:
                        line = line.replace("`", r"\`")
                        line = line.replace("<", "&lt;")
                        line = line.replace(">", "&gt;")
                        line = line.replace(" ", "&nbsp;")
                        line = line.replace("*", "&ast;")
                        line = line.replace("_", "&lowbar;")
                        line = line.replace("-", "&#45;")
                        line = line.replace(".", "&#46;")
                        line = line.replace("!", "&#33;")
                        line = line.replace("(", "&#40;")
                        line = line.replace(")", "&#41;")
                        line = line.replace("$", "&#36;")
                    lines[i] = "<br>" + line
        text = "".join(lines)
        return text

    def _remove_image_special(self, text):
        import re

        text = text.replace('<ref>', '').replace('</ref>', '')
        return re.sub(r'<box>.*?(</box>|$)', '', text)

    def append_user_input(self, query, chatbot, task_history):
        task_text = query
        if len(query) >= 2 and query[-1] in self.PUNCTUATION and query[-2] not in self.PUNCTUATION:
            task_text = query[:-1]
        chatbot = chatbot + [(self._parse_text(query), None)]
        task_history = task_history + [(task_text, None)]
        return '', chatbot, task_history

    def predict(self, chatbot, task_history, top_p, temperature):
        import copy

        chat_query = chatbot[-1][0]
        query = task_history[-1][0]
        history_cp = copy.deepcopy(task_history)
        full_response = ""

        history_filter = []
        pic_idx = 1
        pre = ""
        for _, (q, a) in enumerate(history_cp):
            if isinstance(q, (tuple, list)):  # query is image path
                q = f'Picture {pic_idx}: <img>{q[0]}</img>'
                pre += q + '\n'  # image path as prefix
                pic_idx += 1
            else:  # (query, response) pair; query is text
                pre += q
                history_filter.append((pre, a))  # (image path + text, response)
                pre = ""

        history, message = history_filter[:-1], history_filter[-1][0]

        for response in self.__llm.chat_stream(self.__tokenizer, message, history=history):
            if self.stop_event.is_set():
                break

            chatbot[-1] = (self._parse_text(chat_query), self._remove_image_special(self._parse_text(response)))

            yield chatbot
            full_response = self._parse_text(response)

        self.stop_event.clear()
        response = full_response
        history.append((message, response))
        image = self.__tokenizer.draw_bbox_on_latest_picture(response, history)
        if image is not None:
            temp_dir = secrets.token_hex(20)
            temp_dir = Path(self.__uploaded_file_dir) / temp_dir
            temp_dir.mkdir(exist_ok=True, parents=True)
            name = f"tmp{secrets.token_hex(5)}.jpg"
            filename = temp_dir / name
            image.save(str(filename))
            chatbot.append((None, (str(filename),)))
        else:
            chatbot[-1] = (self._parse_text(chat_query), response)

        task_history[-1] = (query, full_response)
        yield chatbot
