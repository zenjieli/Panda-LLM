import os
import os.path as osp
from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub.constants import HF_HUB_CACHE

CUSTOM_WEIGHTS_DIR = osp.expanduser("~/weights/manual")


def download_file(hf_model_tag: str, filename: str, token: str=None) -> str:
    """Download a file or files from Huggingface Hub. See also https://huggingface.co/docs/huggingface_hub/en/guides/download
    Args:
        hf_model_tag: For example, "repo_id:revision"
        filename (optional): The filename for the file pattern of the file(s) to download, such as my_file.gguf or my_file*.gguf
    Returns:
        str: Downloading result message.
    """

    substrings = hf_model_tag.split(':')
    repo_id, revision = hf_model_tag.split(':') if ':' in hf_model_tag else (substrings[0], None)

    try:
        token = token if token else None

        if filename:
            if "*" in filename:  # Multiplfile download
                subdir = filename.replace("*.", "").replace("*", "")
                result = snapshot_download(repo_id=repo_id, allow_patterns=filename, revision=revision,
                                           local_dir=osp.join(HF_HUB_CACHE, subdir), token=token)
            else:  # Single file download
                result = hf_hub_download(repo_id=repo_id, filename=filename, revision=revision, local_dir=HF_HUB_CACHE, token=token)
        else:
            result = snapshot_download(repo_id=repo_id, cache_dir=HF_HUB_CACHE, revision=revision, token=token)

        # Make the downloaded files accessible to all users
        change_model_permissions(result, model_id=repo_id, mode=0o777)

        return f"Downloaded to {osp.join(os.getcwd(), result)}"
    except Exception as e:
        return f"Downloading failed: {str(e)}"


def change_model_permissions(model_path: str, model_id: str, mode: int = 0o755):
    """
    Recursively change permissions for the model directory
    The input model path is like "/path/to/hf/hub/models--Org--ID/snapshots/hash".
    We need to change the permissions of "/path/to/hf/hub/models--Org--ID".
    """
    from pathlib import Path

    subdir = ""
    items = model_path.split("/")
    model_id = model_id.replace("/", "--")
    for item in items:
        subdir = "/" if item == "" else osp.join(subdir, item)
        if model_id in item:
            break

    core_model_path = Path(subdir)
    core_model_path.chmod(mode)

    # Recursively change permissions for all subdirectories containing model_id and all files
    for item in core_model_path.rglob('*'):
        item.chmod(mode)


def get_cached_model_ids():
    # List all model directories. A dir is like "models--org--model_name"
    model_dirs = [d for d in os.listdir(HF_HUB_CACHE) if d.startswith("models--")]

    # Extract model IDs, like "org/model_name"
    model_ids = ["/".join(d.split("--")[1:]) for d in model_dirs]

    # List all gguf dirs in TRANSFORMERS_CACHE
    gguf_dirs = [d for d in os.listdir(HF_HUB_CACHE) if osp.isdir(
        osp.join(HF_HUB_CACHE, d)) and d.lower().endswith("gguf")]

    # List all gguf files in TRANSFORMERS_CACHE
    gguf_files = [d for d in os.listdir(HF_HUB_CACHE) if osp.isfile(
        osp.join(HF_HUB_CACHE, d)) and d.lower().endswith(".gguf")]

    # List all custom weights (such as the fine-tuned models in house); the model name will look like "custom/my_custom_model"
    if osp.exists(CUSTOM_WEIGHTS_DIR):
        custom_weights = ["custom/" + d for d in os.listdir(CUSTOM_WEIGHTS_DIR) if osp.isdir(osp.join(CUSTOM_WEIGHTS_DIR, d))]
    else:
        custom_weights = []

    return model_ids + gguf_dirs + gguf_files + custom_weights


def get_model_list():
    """Get the list of downloaded models
    """
    items = get_cached_model_ids()
    items.sort(key=str.lower)
    return items
