import torch


def get_gpu_memory_usage() -> str:
    """Get the GPU memory usage (MB) for each GPU in a string
    """
    num_gpus = torch.cuda.device_count()

    # For each GPU, get and print the memory usage
    results = []
    for i in range(num_gpus):
        # Get the current GPU memory usage
        free_mem, total_mem = torch.cuda.mem_get_info(i)

        # Convert bytes to MB
        total_mem_MB = round(total_mem/(1024*1024))
        used_mem_MB = round((total_mem-free_mem)/(1024*1024))

        results.append(f'GPU {i}: {used_mem_MB}MB/{total_mem_MB}MB')

    return '; '.join(results) if len(results) > 0 else ''
