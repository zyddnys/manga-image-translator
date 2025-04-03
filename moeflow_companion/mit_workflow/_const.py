import uuid
import datetime
from pathlib import Path
import functools

_storage_dir = Path(__file__).parent.parent.parent / "storage"
storage_dir = _storage_dir.resolve()


def create_unique_dir(suffix: str | None = None) -> Path:
    if suffix is None:
        suffix = ""
    parts = [
        datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
        uuid.uuid4().hex[0:8],
    ]
    if suffix:
        parts.append(suffix)

    return _storage_dir / "-".join(parts)


@functools.lru_cache(maxsize=1)
def is_cuda_avaiable():
    import torch

    return torch.cuda.is_available()
