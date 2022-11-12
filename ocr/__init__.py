import numpy as np
from typing import List

from utils import Quadrilateral
from .common import CommonOCR, OfflineOCR
from .model_32px import Model32pxOCR
from .model_48px_ctc import Model48pxCTCOCR

OCRS = {
	'32px': Model32pxOCR,
	'48px_ctc': Model48pxCTCOCR,
}
ocr_cache = {}

def get_ocr(key: str, *args, **kwargs) -> CommonOCR:
	if key not in OCRS:
		raise ValueError(f'Could not find OCR for: "{key}". Choose from the following: %s' % ','.join(OCRS))
	if not ocr_cache.get(key):
		ocr = OCRS[key]
		ocr_cache[key] = ocr(*args, **kwargs)
	return ocr_cache[key]

async def prepare(ocr_key: str, use_cuda: bool):
	ocr = get_ocr(ocr_key)
	if isinstance(ocr, OfflineOCR):
		await ocr.download()
		await ocr.load('cuda' if use_cuda else 'cpu')

async def dispatch(ocr_key: str, image: np.ndarray, textlines: List[Quadrilateral], use_cuda: bool = False, verbose: bool = False) -> List[Quadrilateral]:
	ocr = get_ocr(ocr_key)
	if isinstance(ocr, OfflineOCR):
		await ocr.load('cuda' if use_cuda else 'cpu')
	return await ocr.recognize(image, textlines, verbose)
