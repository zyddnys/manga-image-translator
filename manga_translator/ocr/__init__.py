import numpy as np
from typing import List, Optional
from .common import CommonOCR, OfflineOCR
from .model_32px import Model32pxOCR
from .model_48px import Model48pxOCR
from .model_48px_ctc import Model48pxCTCOCR
from .model_manga_ocr import ModelMangaOCR
from ..config import Ocr, OcrConfig
from ..utils import Quadrilateral

OCRS = {
    Ocr.ocr32px: Model32pxOCR,
    Ocr.ocr48px: Model48pxOCR,
    Ocr.ocr48px_ctc: Model48pxCTCOCR,
    Ocr.mocr: ModelMangaOCR,
}
ocr_cache = {}

def get_ocr(key: Ocr, *args, **kwargs) -> CommonOCR:
    if key not in OCRS:
        raise ValueError(f'Could not find OCR for: "{key}". Choose from the following: %s' % ','.join(OCRS))
    if not ocr_cache.get(key):
        ocr = OCRS[key]
        ocr_cache[key] = ocr(*args, **kwargs)
    return ocr_cache[key]

async def prepare(ocr_key: Ocr, device: str = 'cpu'):
    ocr = get_ocr(ocr_key)
    if isinstance(ocr, OfflineOCR):
        await ocr.download()
        await ocr.load(device)

async def dispatch(ocr_key: Ocr, image: np.ndarray, regions: List[Quadrilateral], config:Optional[OcrConfig] = None, device: str = 'cpu', verbose: bool = False) -> List[Quadrilateral]:
    ocr = get_ocr(ocr_key)
    if isinstance(ocr, OfflineOCR):
        await ocr.load(device)
    config = config or OcrConfig()
    return await ocr.recognize(image, regions, config, verbose)

async def unload(ocr_key: Ocr):
    ocr_cache.pop(ocr_key, None)
