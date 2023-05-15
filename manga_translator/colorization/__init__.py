import numpy as np

from .common import CommonColorizer, OfflineColorizer
from .manga_colorization_v2 import MangaColorizationV2

COLORIZERS = {
    'mc2': MangaColorizationV2,
}
colorizer_cache = {}

def get_colorizer(key: str, *args, **kwargs) -> CommonColorizer:
    if key not in COLORIZERS:
        raise ValueError(f'Could not find colorizer for: "{key}". Choose from the following: %s' % ','.join(COLORIZERS))
    if not colorizer_cache.get(key):
        upscaler = COLORIZERS[key]
        colorizer_cache[key] = upscaler(*args, **kwargs)
    return colorizer_cache[key]

async def prepare(key: str):
    upscaler = get_colorizer(key)
    if isinstance(upscaler, OfflineColorizer):
        await upscaler.download()

async def dispatch(key: str, image: np.ndarray, device: str = 'cpu') -> np.ndarray:
    colorizer = get_colorizer(key)
    if isinstance(colorizer, OfflineColorizer):
        await colorizer.load(device)
    return await colorizer.colorize(image)
