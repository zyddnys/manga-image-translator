import functools


@functools.lru_cache(maxsize=1)
def is_cuda_avaiable() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except:
        return False
