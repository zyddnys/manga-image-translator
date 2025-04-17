from typing import Optional

import numpy as np

from .common import CommonInpainter, OfflineInpainter
from .inpainting_aot import AotInpainter
from .inpainting_lama_mpe import LamaMPEInpainter, LamaLargeInpainter
from .inpainting_sd import StableDiffusionInpainter
from .none import NoneInpainter
from .original import OriginalInpainter
from ..config import Inpainter, InpainterConfig

INPAINTERS = {
    Inpainter.default: AotInpainter,
    Inpainter.lama_large: LamaLargeInpainter,
    Inpainter.lama_mpe: LamaMPEInpainter,
    Inpainter.sd: StableDiffusionInpainter,
    Inpainter.none: NoneInpainter,
    Inpainter.original: OriginalInpainter,
}
inpainter_cache = {}

def get_inpainter(key: Inpainter, *args, **kwargs) -> CommonInpainter:
    if key not in INPAINTERS:
        raise ValueError(f'Could not find inpainter for: "{key}". Choose from the following: %s' % ','.join(INPAINTERS))
    if not inpainter_cache.get(key):
        inpainter = INPAINTERS[key]
        inpainter_cache[key] = inpainter(*args, **kwargs)
    return inpainter_cache[key]

async def prepare(inpainter_key: Inpainter, device: str = 'cpu'):
    inpainter = get_inpainter(inpainter_key)
    if isinstance(inpainter, OfflineInpainter):
        await inpainter.download()
        await inpainter.load(device)

async def dispatch(inpainter_key: Inpainter, image: np.ndarray, mask: np.ndarray, config: Optional[InpainterConfig], inpainting_size: int = 1024, device: str = 'cpu', verbose: bool = False) -> np.ndarray:
    inpainter = get_inpainter(inpainter_key)
    if isinstance(inpainter, OfflineInpainter):
        await inpainter.load(device)
    config = config or InpainterConfig()
    return await inpainter.inpaint(image, mask, config, inpainting_size, verbose)

async def unload(inpainter_key: Inpainter):
    inpainter_cache.pop(inpainter_key, None)