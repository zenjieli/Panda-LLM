import os
import os.path as osp
from huggingface_hub import hf_hub_download, snapshot_download
from transformers import TRANSFORMERS_CACHE

CUSTOM_WEIGHTS_DIR = osp.expanduser("~/weights/manual")


def download_file(hf_model_tag: str, filename: str) -> str:
    """Download a file or files from Huggingface Hub.
    Args:
        hf_model_tag: For example, "repo_id:revision"
        filename (optional): The filename for the file pattern of the file(s) to download, such as my_file.gguf or my_file*.gguf
    Returns:
        str: Downloading result message.
    """

    substrings = hf_model_tag.split(':')
    repo_id, revision = hf_model_tag.split(':') if ':' in hf_model_tag else (substrings[0], None)

    try:
        if filename:
            if "*" in filename:  # Multiplfile download
                subdir = filename.replace("*.", "").replace("*", "")
                result = snapshot_download(repo_id=repo_id, allow_patterns=filename, revision=revision,
                                           local_dir=osp.join(TRANSFORMERS_CACHE, subdir))
            else:  # Single file download
                result = hf_hub_download(repo_id=repo_id, filename=filename, revision=revision, local_dir=TRANSFORMERS_CACHE)
        else:
            result = snapshot_download(repo_id=repo_id, cache_dir=TRANSFORMERS_CACHE, revision=revision)

        return f'Downloaded to {osp.join(os.getcwd(), result)}' if osp.exists(result) else result
    except Exception as e:
        return f'Downloading failed: {str(e)}'


def get_cached_model_ids():
    # List all model directories. A dir is like "models--org--model_name"
    model_dirs = [d for d in os.listdir(TRANSFORMERS_CACHE) if d.startswith("models--")]

    # Extract model IDs, like "org/model_name"
    model_ids = ["/".join(d.split("--")[1:]) for d in model_dirs]

    # List all gguf dirs in TRANSFORMERS_CACHE
    gguf_dirs = [d for d in os.listdir(TRANSFORMERS_CACHE) if osp.isdir(
        osp.join(TRANSFORMERS_CACHE, d)) and d.lower().endswith("gguf")]

    # List all gguf files in TRANSFORMERS_CACHE
    gguf_files = [d for d in os.listdir(TRANSFORMERS_CACHE) if osp.isfile(
        osp.join(TRANSFORMERS_CACHE, d)) and d.lower().endswith(".gguf")]

    # List all custom weights (such as the fine-tuned models in house); the model name will look like "custom/my_custom_model"
    custom_weights = ["custom/" + d for d in os.listdir(CUSTOM_WEIGHTS_DIR) if osp.isdir(osp.join(CUSTOM_WEIGHTS_DIR, d))]

    return model_ids + gguf_dirs + gguf_files + custom_weights


def get_model_list():
    """Get the list of downloaded models
    """
    items = get_cached_model_ids()
    items.sort(key=str.lower)
    return items
