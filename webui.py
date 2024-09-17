"""A simple web interactive chat demo based on gradio."""

from argparse import ArgumentParser
from typing import Tuple, List
import gradio as gr
import torch
import os.path as osp
import gc

import utils.text_processing as text_processing
from utils.model_utils import ModelType, get_model_type, get_block_count_from_llama_meta
from modules.AutoModel import AutoModel
from modules.QwenVLModel import QwenVLModel
from modules.LLaVAModel import LLaVAModel
from modules.MiniCPMModel import MiniCPMModel

from utils.download_utils import get_model_list
import utils.ui_utils as ui_utils
from utils.custom_config import CustomConfig
from modules import shared

ROOT_DIR = 'weights/hf'


def update_model_list():
    return gr.Dropdown(choices=get_model_list(ROOT_DIR))


def reset_state(history) -> Tuple[str, str]:
    if history is not None:
        history.clear()
    collect_gabbage()

    return history, ''


def collect_gabbage():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def add_file(history, task_history, file):
    if shared.model.support_image():
        history = history + [((file.name,), None)]
        task_history = task_history + [((file.name,), None)]
        return history, task_history
    else:
        gr.Info('Image not supported.')
        return history, task_history


def on_model_selection_change(model_list_dropdown, n_gpu_layers, n_ctx_1024, lora_path, load_in_8bit):
    model_name = model_list_dropdown

    config = shared.custom_configs.try_get(model_name)
    if config is not None:
        n_gpu_layers = config.n_gpu_layers
        n_ctx_1024 = config.n_ctx_1024
        lora_path = config.lora_path
        load_in_8bit = config.load_in_8bit

    return n_gpu_layers, n_ctx_1024, lora_path, load_in_8bit


def load_model(model_list_dropdown, n_gpu_layers, n_ctx, lora_path, load_in_8bit) -> Tuple[str, str]:
    """
    Parameters:
        model_list_dropdown: model name
        n_gpu_layers: number of layers to offload to GPU. Only supported by llama-cpp for now
        n_ctx: context window size (unit: 1024 tokens). Only supported by llama-cpp for now
    """
    from utils.gpu_utils import get_gpu_memory_usage

    if shared.model is not None:
        shared.model = None
        gc.collect()
        torch.cuda.empty_cache()
    else:
        print('shared.model is None')

    lora_name = lora_path.split('/')[-1] if lora_path else ''
    model_description = model_list_dropdown + (f' (LORA: {lora_name})' if lora_path else "")
    model_path = osp.join(ROOT_DIR, model_list_dropdown)
    model_type = get_model_type(model_path)
    meta_info = ''

    if model_type in [ModelType.GPTQ, ModelType.Other]:
        shared.model = AutoModel(model_path, lora_path=lora_path, load_in_8bit=load_in_8bit)
    elif model_type == ModelType.GGUF:
        from modules.GGUFModel import GGUFModel
        shared.model = GGUFModel(model_path, gpu_layers=n_gpu_layers, n_ctx=n_ctx * 1024)
        block_count = get_block_count_from_llama_meta(shared.model.core_model.metadata)
        meta_info = f'block count {block_count}'
    elif model_type == ModelType.LLaVA:
        shared.model = LLaVAModel(model_path, gpu_layers=n_gpu_layers, n_ctx=n_ctx * 1024)
    elif model_type == ModelType.MiniCPM:
        shared.model = MiniCPMModel(model_path, load_in_8bit=load_in_8bit)
    elif model_type == ModelType.QWEN_VL:
        shared.model = QwenVLModel(model_path)
    else:
        raise NotImplementedError(f'Unsupported model type: {model_type}')

    # Get the number of parameters in shared.model
    num_params = sum(p.numel() for p in shared.model.core_model.parameters())

    return f'Model loaded: {model_description} ' + (f' ({meta_info})' if meta_info else '') + \
        f' Parameters: {num_params/1024/1024/1024:.1f}B', get_gpu_memory_usage()


def save_custom_config(model_list_dropdown, n_gpu_layers, n_ctx_1024, lora_path, load_in_8bit):
    """Save custom model config. See also `load_model`
    """
    shared.custom_configs.add(CustomConfig(model_list_dropdown, n_gpu_layers, n_ctx_1024, lora_path, load_in_8bit))
    shared.custom_configs.save_to_json()


def append_user_input(query, chatbot, system_prompt) -> Tuple[str, List, str]:
    if not shared.model:
        return query, chatbot, 'No model loaded'
    else:
        query, chatbot = shared.model.append_user_input(query, chatbot)

        # Compute the token count
        tokens = shared.model.try_tokenize(chatbot, system_prompt)
        token_count = len(tokens) if tokens is not None else 0
        prediction_status = f'Token count: {token_count}' if token_count > 0 else ''
        if not shared.model.check_token_count(token_count):
            prediction_status += ' (too many tokens)'
            chatbot.pop()

        return query, chatbot, prediction_status


def update_prediction_status(chatbot):
    return str(len(chatbot))


def predict(chatbot, system_prompt, top_k, top_p, temperature, enable_postprocessing):
    if shared.model:
        params = {'enable_postprcessing': enable_postprocessing}
        if system_prompt:
            params['system_prompt'] = system_prompt
        if top_k > 0:
            params['top_k'] = top_k
        if top_p > 0:
            params['top_p'] = top_p
        if temperature > 0:
            params['temperature'] = temperature

        yield from shared.model.predict(chatbot, params)
    else:
        yield chatbot


def set_stop_event():
    shared.model.stop_event.set()


def on_chatbot_change(chatbot):
    if not chatbot or len(chatbot) == 0:
        print('Chatbot is emptied: ')


def main(args):
    with gr.Blocks(title='Panda Chatbot') as demo:
        gr.Markdown(ui_utils.model_text(''))
        model_param_elements = {}

        with gr.Tab('Main'):
            chatbot = gr.Chatbot()
            chatbot.change(on_chatbot_change, chatbot)

            with gr.Row():
                with gr.Column(scale=4):
                    user_input = gr.Textbox(show_label=False, placeholder='Input...', container=False)
                    task_history = gr.State([])
                    model_param_elements['task_history'] = task_history
                    with gr.Row():
                        submit_btn = gr.Button('🚀 Submit')
                        regen_btn = gr.Button('🔁 Regenerate')
                        pop_last_message_btn = gr.Button('🧹 Remove latest')
                        stop_btn = gr.Button('🛑 Stop')
                    with gr.Row():
                        prediction_status_label = gr.Markdown()
                        model_param_elements['enable_postprocessing'] = gr.Checkbox(True, label="Postprocess output")
                with gr.Column(scale=1):
                    empty_btn = gr.Button('🗑️ Clear History')
                    addfile_btn = gr.UploadButton("🖼️ Image...", file_types=["image"])
                    addfile_btn.upload(add_file, [chatbot, task_history, addfile_btn],
                                       [chatbot, task_history], show_progress=True)

                    model_param_elements['top_k'] = gr.Slider(-1, 50, value=-1, step=1, label='Top K')
                    model_param_elements['top_p'] = gr.Slider(0, 1, value=0.8, step=0.01, label='Top P')
                    model_param_elements['temperature'] = gr.Slider(0.01, 1, value=0.2, step=0.01, label='Temperature')

        with gr.Tab('Models'):
            with gr.Row():
                with gr.Column(scale=5):
                    model_list_dropdown = gr.Dropdown(get_model_list(ROOT_DIR), label='Models', interactive=True)
                    lora_path = gr.Textbox(placeholder="Path to Lora model (For transformers library models)", value='',
                                           show_label=False, max_lines=1, container=False)
                    with gr.Row():
                        model_refresh_btn = gr.Button('🔃 Refresh')
                        model_refresh_btn.click(lambda: gr.Dropdown(get_model_list(ROOT_DIR)),
                                                inputs=[], outputs=model_list_dropdown)
                        model_load_btn = gr.Button('🚚 Load')
                    model_save_btn = gr.Button('🗄 Save User Config')
                    with gr.Row():
                        gpu_layers_slider = gr.Slider(label='GPU layers to offload to GPU', info='For GGUF', value=-1,
                                                      minimum=-1, maximum=200, step=1, interactive=True)
                        ctx_length_slider = gr.Slider(label='Context window length (K)', info='For GGUF', value=1,
                                                      minimum=1, maximum=32, step=1, interactive=True)
                    with gr.Row():
                        load_in_8bit_checkbox = gr.Checkbox(label='Load in 8-bit', info='For transformers library models')

                    model_param_elements['system_prompt'] = gr.Textbox(
                        show_label=False, placeholder='System prompt...', container=False)

                with gr.Column(scale=5):
                    from utils.download_utils import download_file
                    # See https://huggingface.co/docs/huggingface_hub/en/guides/download
                    hf_model_tag = gr.Textbox(label="Download model", info=ui_utils.DOWNLOAD_MODEL_INSTRUCTION,
                                              show_label=True, container=False)
                    hf_filename = gr.Textbox(placeholder="File name (for GGUF models)",
                                             show_label=False, max_lines=1, container=False)
                    download_btn = gr.Button('Download')
                    model_status_label = gr.Markdown()
                    gpu_usage_label = gr.Markdown()

            download_btn.click(download_file, inputs=[hf_model_tag, hf_filename], outputs=[model_status_label])
            model_load_btn.click(load_model,
                                 inputs=[model_list_dropdown, gpu_layers_slider,
                                         ctx_length_slider, lora_path, load_in_8bit_checkbox],
                                 outputs=[model_status_label, gpu_usage_label])
            model_save_btn.click(save_custom_config,
                                 inputs=[model_list_dropdown, gpu_layers_slider, ctx_length_slider, lora_path, load_in_8bit_checkbox])
            model_list_dropdown.change(on_model_selection_change,
                                       [model_list_dropdown, gpu_layers_slider, ctx_length_slider, lora_path, load_in_8bit_checkbox],
                                       [gpu_layers_slider, ctx_length_slider, lora_path, load_in_8bit_checkbox])

        gr.Markdown(ui_utils.SUPPORTED_MODELS_TEXT)

        predict_params = [chatbot, model_param_elements['system_prompt'], model_param_elements['top_k'], model_param_elements['top_p'],
                          model_param_elements['temperature'], model_param_elements['enable_postprocessing']]
        submit_btn.click(append_user_input, [user_input, chatbot, model_param_elements['system_prompt']],
                         [user_input, chatbot, prediction_status_label]).then(predict, predict_params, chatbot)

        # Same as "Submit"; trigged when user presses enter
        user_input.submit(append_user_input, [user_input, chatbot, model_param_elements['system_prompt']],
                          [user_input, chatbot, prediction_status_label]).then(
            predict, predict_params, [chatbot])
        regen_btn.click(text_processing.remove_last_reply, [chatbot], chatbot).then(
            predict, predict_params, chatbot)
        pop_last_message_btn.click(text_processing.remove_last_message, [chatbot], chatbot)
        stop_btn.click(set_stop_event, queue=False)
        empty_btn.click(reset_state, chatbot, [chatbot, prediction_status_label], queue=False)

    demo.queue()
    demo.launch(server_name=args.server_name, server_port=args.server_port, inbrowser=False)


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--server-name', type=str, default='127.0.0.1', help='Demo server name')
    parser.add_argument('--server-port', type=int, default=8111, help='Demo server port')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    main(args)
