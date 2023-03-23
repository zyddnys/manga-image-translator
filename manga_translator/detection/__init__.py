import numpy as np
from typing import List, Tuple

from .default import DefaultDetector
from .ctd import ComicTextDetector
from .common import CommonDetector, OfflineDetector
from ..utils import TextBlock

DETECTORS = {
    'default': DefaultDetector,
    'ctd': ComicTextDetector,
}
detector_cache = {}

def get_detector(key: str, *args, **kwargs) -> CommonDetector:
    if key not in DETECTORS:
        raise ValueError(f'Could not find detector for: "{key}". Choose from the following: %s' % ','.join(DETECTORS))
    if not detector_cache.get(key):
        detector = DETECTORS[key]
        detector_cache[key] = detector(*args, **kwargs)
    return detector_cache[key]

async def prepare(detector_key: str):
    detector = get_detector(detector_key)
    if isinstance(detector, OfflineDetector):
        await detector.download()

async def dispatch(detector_key: str, image: np.ndarray, detect_size: int, text_threshold: float, box_threshold: float, unclip_ratio: float,
                   invert: bool, gamma_correct: bool, rotate: bool, auto_rotate: bool = False, device: str = 'cpu', verbose: bool = False) -> Tuple[List[TextBlock], np.ndarray]:
    detector = get_detector(detector_key)
    if isinstance(detector, OfflineDetector):
        await detector.load(device)
    return await detector.detect(image, detect_size, text_threshold, box_threshold, unclip_ratio, invert, gamma_correct, rotate, auto_rotate, verbose)
