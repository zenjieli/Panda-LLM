"""A simple web interactive chat demo based on gradio."""

from argparse import ArgumentParser
import gradio as gr
import torch
from transformers import AutoConfig

import utils.text_processing as text_processing
from utils.model_utils import ModelType, get_model_type
from modules.GPTQModel import GPTQModel
from modules.GGUFModel import GGUFModel
from modules.QwenVLModel import QwenVLModel


def reset_state(history):
    history.clear()
    collect_gabbage()

    return history


def collect_gabbage():
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def add_file(history, task_history, file):
        history = history + [((file.name,), None)]
        task_history = task_history + [((file.name,), None)]
        return history, task_history

def main(args):
    import utils.ui_utils as ui_utils

    with gr.Blocks(title='Panda Chatbot') as demo:
        gr.Markdown(ui_utils.HEADER_TEXT)
        gr.Markdown(ui_utils.model_text(args.model_path))

        chatbot = gr.Chatbot()

        with gr.Row():
            with gr.Column(scale=4):
                user_input = gr.Textbox(show_label=False, placeholder='Input...', container=False)
                task_history = gr.State([])
                with gr.Row():
                    submit_btn = gr.Button('🚀 Submit')
                    regen_btn = gr.Button('🔁 Regenerate')
                    pop_last_message_btn = gr.Button('🧹 Remove latest')
                    stop_btn = gr.Button('🛑 Stop')
            with gr.Column(scale=1):
                emptyBtn = gr.Button('🗑️ Clear History')
                if model.support_image():
                    addfile_btn = gr.UploadButton("🖼️ Image...", file_types=["image"])
                    addfile_btn.upload(add_file, [chatbot, task_history, addfile_btn], [chatbot, task_history], show_progress=True)

                top_p_slider = gr.Slider(0, 1, value=0.8, step=0.01, label='Top P')
                temperature_slider = gr.Slider(0.01, 1, value=0.2, step=0.01, label='Temperature')

        gr.Markdown(ui_utils.SUPPORTED_MODELS_TEXT)

        submit_btn.click(model.append_user_input, [user_input, chatbot, task_history], [user_input, chatbot, task_history], queue=True).then(
            model.predict, [chatbot, task_history, top_p_slider, temperature_slider], chatbot)

        # Same as "Submit"; trigged when user presses enter
        user_input.submit(model.append_user_input, [user_input, chatbot, task_history], [user_input, chatbot, task_history], queue=False).then(
            model.predict, [chatbot, task_history, top_p_slider, temperature_slider], chatbot)
        regen_btn.click(text_processing.remove_last_reply, [chatbot], chatbot, queue=True).then(
            model.predict, [chatbot, task_history, top_p_slider, temperature_slider], chatbot)
        pop_last_message_btn.click(text_processing.remove_last_message, [chatbot], chatbot, queue=True)
        stop_btn.click(lambda: model.stop_event.set(), queue=False)
        emptyBtn.click(reset_state, chatbot, chatbot, queue=False)

    demo.queue()
    demo.launch(server_name=args.server_name, server_port=args.server_port, inbrowser=False)


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--model-path', type=str)
    parser.add_argument('--gguf-gpu-layers', type=int, default=0, help='The number of model layers on GPU')
    parser.add_argument('--server-name', type=str, default='127.0.0.1', help='Demo server name')
    parser.add_argument('--server-port', type=int, default=8111, help='Demo server port')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    model_type = get_model_type(args.model_path)

    model = None
    if model_type in [ModelType.GPTQ, ModelType.Other]:
        model = GPTQModel(args.model_path)
    elif model_type == ModelType.GGUF:
        model = GGUFModel(args.model_path, gpu_layers=args.gguf_gpu_layers)
    elif model_type == ModelType.QWEN_VL:
        model = QwenVLModel(args.model_path)
    else:
        raise NotImplementedError(f'Unsupported model type: {model_type}')

    main(args)
