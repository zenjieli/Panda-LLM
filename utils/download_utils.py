import os
import os.path as osp
from huggingface_hub import hf_hub_download, snapshot_download


def download_file(hf_model_tag, filename):
    # repo_id is the substring of hf_model_tag before :, revision is the substring after :
    substrings = hf_model_tag.split(':')
    repo_id, revision = hf_model_tag.split(':') if ':' in hf_model_tag else (substrings[0], None)

    try:
        if filename:
            result = hf_hub_download(repo_id=repo_id, filename=filename, revision=revision)
        else:
            result = snapshot_download(repo_id=repo_id, revision=revision)

        return f'Downloaded to {osp.join(os.getcwd(), result)}' if osp.exists(result) else result
    except Exception as e:
        return f'Downloading failed: {str(e)}'

def get_cached_model_ids():
    from transformers import TRANSFORMERS_CACHE

    # List all model directories. A dir is like "models--org--model_name"
    model_dirs = [d for d in os.listdir(TRANSFORMERS_CACHE) if d.startswith("models--")]

    # Extract model IDs, like "org/model_name"
    return ["/".join(d.split("--")[1:]) for d in model_dirs]

def get_model_list(root_dir):
    # get all the subdirectories and files in root_dir
    items = get_cached_model_ids()
    if osp.exists(root_dir):
        items.extend([osp.basename(item) for item in os.listdir(root_dir) if not item.startswith('.')])
    items.sort(key=str.lower)
    return items
