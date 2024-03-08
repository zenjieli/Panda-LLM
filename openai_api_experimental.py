"""
LLM as a server with OpenAI API
"""
from argparse import ArgumentParser
from contextlib import asynccontextmanager
from typing import Dict

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, TextIteratorStreamer, StoppingCriteriaList
from transformers.generation import GenerationConfig

from threading import Thread

from utils.model_utils import ModelType, get_model_type, get_block_count_from_llama_meta
import modules.shared as shared
from modules.BaseModel import BaseModel
from modules.GGUFModel import GGUFModel

from utils.openai_types import (
    ModelCard,
    ModelList,
    DeltaMessage,
    ChatMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseStreamChoice,
    ChatCompletionResponseChoice)


def _gc(forced: bool = False):
    global args
    if args.disable_gc and not forced:
        return

    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@asynccontextmanager
async def lifespan(app: FastAPI):  # collects GPU memory
    yield
    _gc(forced=True)


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)


@app.get('/v1/models', response_model=ModelList)
async def list_models():
    global model_args
    model_card = ModelCard(id='Qwen1.5')
    return ModelList(data=[model_card])


# To work around that unpleasant leading-\n tokenization issue!
def add_extra_stop_words(stop_words):
    if stop_words:
        _stop_words = []
        _stop_words.extend(stop_words)
        for x in stop_words:
            s = x.lstrip('\n')
            if s and (s not in _stop_words):
                _stop_words.append(s)
        return _stop_words
    return stop_words


def trim_stop_words(response, stop_words):
    if stop_words:
        for stop in stop_words:
            idx = response.find(stop)
            if idx != -1:
                response = response[:idx]
    return response


def parse_messages(messages):
    system = messages[0].content.strip() if messages[0].role == 'system' else ''

    if messages[-1].role == 'user':
        query = messages[-1].content
    else:
        raise HTTPException(status_code=400, detail='Invalid request: query missing')

    return query, system


@app.post('/v1/chat/completions', response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    stop = StopOnTokens()
    # gen_kwargs = {'stopping_criteria': StoppingCriteriaList([stop]),
    #               'max_new_tokens': 2048,}
    gen_kwargs = {}

    if request.top_k is not None:
        gen_kwargs['top_k'] = request.top_k
    if request.temperature:
        if request.temperature < 0.01:
            gen_kwargs['top_k'] = 1  # greedy decoding
        else:
            # Not recommended. Please tune top_p instead.
            gen_kwargs['temperature'] = request.temperature

    if request.top_p:
        gen_kwargs['top_p'] = request.top_p

    if request.top_k or request.temperature or request.top_p:
        gen_kwargs['do_sample'] = True

    query, system = parse_messages(request.messages)

    if request.stream:
        generate = predict(query, system, request.model, gen_kwargs)  # Coroutine object
        return EventSourceResponse(generate)
    else:
        return predict_nostream(query, system, request.model, gen_kwargs)


def _dump_json(data: BaseModel, *args, **kwargs) -> str:
    try:
        return data.model_dump_json(*args, **kwargs)
    except AttributeError:  # pydantic<2.0.0
        return data.json(*args, **kwargs)  # noqa


class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # Add custom stop conditions here
        # We can also count the numbe of generated tokens

        return False


async def predict(query: str, system: str, model_id: str, gen_kwargs: Dict):
    messages = [{'role': 'system', 'content': system}] if system else []
    messages.append({'role': 'user', 'content': query})
    model_inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                                 tokenize=True, return_tensors='pt').to('cuda')
    streamer = TextIteratorStreamer(tokenizer, timeout=60, skip_prompt=True, skip_special_tokens=True)
    gen_kwargs = {**gen_kwargs,
                  'input_ids': model_inputs,
                  'streamer': streamer}
    t = Thread(target=model.generate, kwargs=gen_kwargs)
    t.start()

    for new_token in streamer:
        if not new_token:
            continue

        choice_data = ChatCompletionResponseStreamChoice(
            index=0, delta=DeltaMessage(content=new_token), finish_reason=None)
        chunk = ChatCompletionResponse(model=model_id,
                                       choices=[choice_data],
                                       object='chat.completion.chunk')
        yield '{}'.format(_dump_json(chunk, exclude_unset=True))

    t.join()


def predict_nostream(query: str, system: str, model_id: str, gen_kwargs: Dict):
    llm = shared.model.llm
    messages = BaseModel.chatbot_to_messages(chatbot=[(query, '')], system_prompt=system)
    reply = llm.create_chat_completion(messages=messages, stream=False, **gen_kwargs)
    reply_msg = reply['choices'][0]['message']['content']

    choice_data = ChatCompletionResponseChoice(index=0, message=ChatMessage(
        role='assistant', content=reply_msg), finish_reason='stop')
    return ChatCompletionResponse(model=model_id,
                                   object='chat.completion',
                                   choices=[choice_data])


def _get_args():
    parser = ArgumentParser()
    parser.add_argument(
        '-c',
        '--checkpoint-path',
        type=str,
        help='Checkpoint name or path, default to %(default)r',
    )
    parser.add_argument('--server-port',
                        type=int,
                        default=8000,
                        help='Demo server port.')
    parser.add_argument(
        '--server-name',
        type=str,
        default='127.0.0.1',
        help=' If you want other computers to access your server, use 0.0.0.0 instead.',
    )
    parser.add_argument(
        '--disable-gc',
        action='store_true',
        help='Disable GC after each response generated.',
    )

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = _get_args()

    model_type = get_model_type(args.checkpoint_path)

    if model_type in [ModelType.GPTQ, ModelType.Other]:
        raise ValueError('Model type {} is not supported'.format(model_type))
    elif model_type == ModelType.GGUF:
        shared.model = GGUFModel(args.checkpoint_path, gpu_layers=-1, n_ctx=2 * 1024)

    uvicorn.run(app, host=args.server_name, port=args.server_port, workers=1)
