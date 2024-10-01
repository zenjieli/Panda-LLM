from argparse import ArgumentParser
import gradio as gr
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

import utils.text_processing as text_processing
from utils.model_utils import ModelType, get_model_type
from modules.auto_model import AutoModel
from modules.gguf_model import GGUFModel


def _parse_text(text):
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

def _remove_image_special(text):
    text = text.replace('<ref>', '').replace('</ref>', '')
    return re.sub(r'<box>.*?(</box>|$)', '', text)

def main():
    model_path = 'models/Qwen-VL-Chat-Int4'

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, resume_download=True, revision='master')

    device_map = "cuda"
    model = AutoModelForCausalLM.from_pretrained('models/Qwen-VL-Chat-Int4', device_map=device_map, trust_remote_code=True).eval()

if __name__ == '__main__':
    main()
