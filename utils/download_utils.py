import os
import os.path as osp
from huggingface_hub import hf_hub_download, snapshot_download


def download_file(hf_model_tag, filename):
    local_dir = 'models/hf'

    # repo_id is the substring of hf_model_tag before :, revision is the substring after :
    substrings = hf_model_tag.split(':')
    repo_id, revision = hf_model_tag.split(':') if ':' in hf_model_tag else (substrings[0], None)
    subdir = '' if filename else repo_id.split('/')[-1]

    try:
        if filename:
            result = hf_hub_download(repo_id=repo_id, filename=filename, revision=revision,
                                     local_dir=local_dir, local_dir_use_symlinks=False)
        else:
            result = snapshot_download(repo_id=repo_id, revision=revision, local_dir=osp.join(local_dir, subdir), local_dir_use_symlinks=False)

        return f'Downloaded to {osp.join(os.getcwd(), result)}' if osp.exists(result) else result
    except Exception as e:
        return f'Downloading failed: {str(e)}'


def get_model_list(root_dir):
    # get all the subdirectories and files in root_dir
    items = [osp.basename(item) for item in os.listdir(root_dir) if not item.startswith('.')]
    items.sort(key=str.lower)
    return items
