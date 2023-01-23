import numpy as np

from .common import CommonInpainter, OfflineInpainter
from .inpainting_aot import AotInpainter
from .inpainting_lama_mpe import LamaMPEInpainter
from .none import NoneInpainter

INPAINTERS = {
    'none': NoneInpainter,
    'default': AotInpainter,
    'lama_mpe': LamaMPEInpainter,
}
inpainter_cache = {}

def get_inpainter(key: str, *args, **kwargs) -> CommonInpainter:
    if key not in INPAINTERS:
        raise ValueError(f'Could not find inpainter for: "{key}". Choose from the following: %s' % ','.join(INPAINTERS))
    if not inpainter_cache.get(key):
        inpainter = INPAINTERS[key]
        inpainter_cache[key] = inpainter(*args, **kwargs)
    return inpainter_cache[key]

async def prepare(inpainter_key: str, use_cuda: bool):
    inpainter = get_inpainter(inpainter_key)
    if isinstance(inpainter, OfflineInpainter):
        await inpainter.download()
        await inpainter.load('cuda' if use_cuda else 'cpu')

async def dispatch(inpainter_key: str, image: np.ndarray, mask: np.ndarray, inpainting_size: int = 1024, use_cuda: bool = False, verbose: bool = False) -> np.ndarray:
    inpainter = get_inpainter(inpainter_key)
    if isinstance(inpainter, OfflineInpainter):
        await inpainter.load('cuda' if use_cuda else 'cpu')
    return await inpainter.inpaint(image, mask, inpainting_size, verbose)
