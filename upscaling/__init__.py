from typing import List
from PIL import Image

from .common import CommonUpscaler, OfflineUpscaler
from .waifu2x import Waifu2xUpscaler

UPSCALERS = {
    'waifu2x': Waifu2xUpscaler,
}
upscaler_cache = {}

def get_upscaler(key: str, *args, **kwargs) -> CommonUpscaler:
    if key not in UPSCALERS:
        raise ValueError(f'Could not find upscaler for: "{key}". Choose from the following: %s' % ','.join(UPSCALERS))
    if not upscaler_cache.get(key):
        upscaler = UPSCALERS[key]
        upscaler_cache[key] = upscaler(*args, **kwargs)
    return upscaler_cache[key]

async def prepare(upscaler_key: str, upscale_ratio: bool):
    if upscale_ratio == 1:
        return

    upscaler = get_upscaler(upscaler_key)
    if isinstance(upscaler, OfflineUpscaler):
        await upscaler.download()

async def dispatch(upscaler_key: str, image_batch: List[Image.Image], upscale_ratio: int, use_cuda: bool = False) -> List[Image.Image]:
    if upscale_ratio == 1:
        return image_batch

    upscaler = get_upscaler(upscaler_key)
    if isinstance(upscaler, OfflineUpscaler):
        await upscaler.load('cuda' if use_cuda else 'cpu')
    return await upscaler.upscale(image_batch, upscale_ratio)
