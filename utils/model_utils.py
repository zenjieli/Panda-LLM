def get_block_count_from_llama_meta(llama_meta: dict) -> int:
    # Find the key ending with '.block_count'
    for key in llama_meta.keys():
        if key.endswith('.block_count'):
            return int(llama_meta[key])

    return -1