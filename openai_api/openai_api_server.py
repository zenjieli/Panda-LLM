"""
LLM as a server with OpenAI API
"""
import os
import os.path as osp
import sys
import base64
import io
from argparse import ArgumentParser
from contextlib import asynccontextmanager
from typing import Dict
from PIL import Image

import torch
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from uuid import uuid4
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import StoppingCriteria


from openai_types import (
    ChatMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice)

# Add parent dir to path
sys.path.append(osp.join(osp.dirname(osp.abspath(__file__)), "../"))

# Ensure models are registered
from modules.auto_model import AutoModel
from modules.model_factory import ModelFactory
from modules import all_models


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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# To work around that unpleasant leading-\n tokenization issue!
def add_extra_stop_words(stop_words):
    if stop_words:
        _stop_words = []
        _stop_words.extend(stop_words)
        for x in stop_words:
            s = x.lstrip("\n")
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
    system = messages[0].content.strip() if messages[0].role == "system" else ""

    query = None
    system = None
    if messages[-1].role == "user":
        content = messages[-1].content
        if isinstance(content, str):
            query = content # Handle text-only content
        elif isinstance(content, list): # Handle multimodal content
            for item in content:
                if item.type == "text":
                    query = item.text
                elif item.type == "image_url":
                    image_url = item.image_url.url
                else:
                    raise HTTPException(status_code=400, detail="Invalid content type")
    else:
        raise HTTPException(status_code=400, detail="Invalid request: query missing")

    return query, system, image_url

# Helper function to process image from base64
def process_image(image_data: str) -> Image.Image:
    try:
        # Remove data URI prefix if present
        if image_data.startswith("data:image"):
            image_data = image_data.split(",")[1]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    global model, tokenizer

    gen_kwargs = {
        "max_new_tokens": request.max_tokens,       
        "do_sample": request.temperature is not None and request.temperature > 0
    }
    if request.top_k is not None:
        gen_kwargs["top_k"] = request.top_k
    if request.temperature is not None:
        gen_kwargs["temperature"] = request.temperature
    if request.top_p:
        gen_kwargs["top_p"] = request.top_p

    query, system, image_url = parse_messages(request.messages)

    if image_url is not None:
        gen_kwargs["image_url"] = image_url
    
    if request.seed is not None: # Handle seed for reproducibility
        torch.manual_seed(request.seed)

    if request.stream:
        print("Streaming is not supported. Fall back to non-streaming mode.")

    return predict_nostream(query, system, request.model, gen_kwargs)

def cleanup_file(file_path: str):
    try:
        os.remove(file_path)
    except Exception as e:
        print(f"Failed to delete file {file_path}: {str(e)}")

@app.post("/v1/files")
async def upload_file(file: UploadFile = File(...), purpose: str = "assistants"):
    file_id = str(uuid4()) # Generate a unique filename from UUID
    file_ext = osp.splitext(file.filename)[-1]

    # Save the file to the server
    filename = f"{file_id}{file_ext}"
    file_location = os.path.join(UPLOAD_DIR, filename)
    try:
        with open(file_location, "wb") as buffer:
            buffer.write(await file.read())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

    # Return metadata about the uploaded file
    return JSONResponse(
        content={
            "id": file_id,
            "filename": filename,
            "purpose": purpose
        },
        status_code=201,
    )


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


def predict_nostream(query: str, system: str, model_id: str, gen_kwargs: Dict):
    global model

    reply = model.predict_simple_nostream(query, system, gen_kwargs)

    choice_data = ChatCompletionResponseChoice(index=0, message=ChatMessage(
        role="assistant", content=reply), finish_reason="stop")
    return ChatCompletionResponse(model=model_id,
                                  object="chat.completion",
                                  choices=[choice_data])


def _get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Checkpoint name or path, default to %(default)r",
    )
    parser.add_argument("--server-port",
                        type=int,
                        default=8112,
                        help="Demo server port.")
    parser.add_argument(
        "--server-name",
        type=str,
        default="0.0.0.0",
        help=" If you want other computers to access your server, use 0.0.0.0.",
    )
    parser.add_argument(
        "--disable-gc",
        action="store_true",
        help="Disable GC after each response generated.",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = _get_args()

    model_class = ModelFactory.get_model_class(args.model_name_or_path)
    if model_class == None:
        model_class = AutoModel

    UPLOAD_DIR = "files" # For uploaded files
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    kwargs = {"gpu_layers": -1, "n_ctx": 4 * 1024}
    model = model_class(args.model_name_or_path, **kwargs)

    uvicorn.run(app, host=args.server_name, port=args.server_port, workers=1)
