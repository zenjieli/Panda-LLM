"""A simple web interactive chat demo based on gradio."""

from argparse import ArgumentParser
import gradio as gr
import torch
from transformers import AutoConfig

import utils.text_processing as text_processing
from utils.model_utils import ModelType, get_model_type
from modules.GPTQModel import GPTQModel
from modules.GGUFModel import GGUFModel


def reset_state(history):
    history.clear()
    collect_gabbage()

    return history


def collect_gabbage():
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def append_user_input(query, history):
    if history is None:
        history = []

    if query != '':
        return '', history + [[text_processing.parse_text(query), '']]
    else:
        return '', history


def main(args):
    import utils.ui_utils as ui_utils

    with gr.Blocks(title='Panda Chatbot') as demo:
        gr.Markdown(ui_utils.HEADER_TEXT)
        gr.Markdown(ui_utils.model_text(args.model_path))

        chatbot = gr.Chatbot()

        with gr.Row():
            with gr.Column(scale=4):
                user_input = gr.Textbox(show_label=False, placeholder='Input...', container=False)
                with gr.Row():
                    submit_btn = gr.Button('🚀 Submit')
                    regen_btn = gr.Button('🔁 Regenerate')
                    pop_last_message_btn = gr.Button('🧹 Remove latest')
                    stop_btn = gr.Button('🛑 Stop')
            with gr.Column(scale=1):
                emptyBtn = gr.Button('🗑️ Clear History')
                top_p_slider = gr.Slider(0, 1, value=0.8, step=0.01, label='Top P')
                temperature_slider = gr.Slider(0.01, 1, value=0.2, step=0.01, label='Temperature')

        gr.Markdown(ui_utils.SUPPORTED_MODELS_TEXT)

        submit_btn.click(append_user_input, [user_input, chatbot], [user_input, chatbot], queue=True).then(
            model.predict, [chatbot, top_p_slider, temperature_slider], chatbot)

        # Same as "Submit"; trigged when user presses enter
        user_input.submit(append_user_input, [user_input, chatbot], [user_input, chatbot], queue=False).then(
            model.predict, [chatbot, top_p_slider, temperature_slider], chatbot)

        regen_btn.click(text_processing.remove_last_reply, [chatbot], chatbot, queue=True).then(
            model.predict, [chatbot, top_p_slider, temperature_slider], chatbot)

        pop_last_message_btn.click(text_processing.remove_last_message, [chatbot], chatbot, queue=True)

        stop_btn.click(lambda: model.stop_event.set(), queue=False)

        emptyBtn.click(reset_state, chatbot, chatbot, queue=False)

    demo.queue()
    demo.launch(server_name=args.server_name, server_port=8111, inbrowser=False, share=args.share)


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--model-path', type=str)
    parser.add_argument('--gguf-gpu-layers', type=int, default=0, help='The number of model layers on GPU')
    parser.add_argument('--share', action='store_true', default=False, help='Create a publicly shareable link')
    parser.add_argument('--server-name', type=str, default='127.0.0.1', help='Demo server name')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    model_type = get_model_type(args.model_path)

    model = None
    if model_type in [ModelType.GPTQ, ModelType.Other]:
        model = GPTQModel(args.model_path)
    elif model_type == ModelType.GGUF:
        model = GGUFModel(args.model_path, gpu_layers=args.gguf_gpu_layers)
    else:
        raise NotImplementedError(f'Unsupported model type: {model_type}')


    main(args)
