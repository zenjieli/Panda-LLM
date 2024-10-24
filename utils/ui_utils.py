import os.path as osp

DOWNLOAD_MODEL_INSTRUCTION = 'Hugging Face username/model path, for instance: facebook/galactica-125m. To specify a branch, add it at the end after a \":\" character like this: facebook/galactica-125m:main.'


def model_icon():
    return f"""\
        <p align="center">
            <img src="https://upload.wikimedia.org/wikipedia/commons/f/fc/Creative-Tail-Animal-panda.svg" style="height: 40px"/></td>
        </p>"""


def supported_models_text(all_model_descriptions: list[str]) -> str:
    return f"<center><font size=2><b>Supported models</b>: Huggingface LLMs, GGUF models, {', '.join(all_model_descriptions)}</center>"
