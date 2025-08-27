import numpy as np
import logging

from .default import DefaultDetector
from .dbnet_convnext import DBConvNextDetector
from .ctd import ComicTextDetector
from .craft import CRAFTDetector
from .paddle import PaddleDetector
from .none import NoneDetector
from .switch import SwitchDetector
from .common import CommonDetector, OfflineDetector
from ..config import Detector

logger = logging.getLogger('manga_translator')

DETECTORS = {
    Detector.default: DefaultDetector,
    Detector.dbconvnext: DBConvNextDetector,
    Detector.ctd: ComicTextDetector,
    Detector.craft: CRAFTDetector,
    Detector.paddle: PaddleDetector,
    Detector.none: NoneDetector,
    Detector.switch: SwitchDetector,
}
detector_cache = {}

def get_detector(key: Detector, *args, **kwargs) -> CommonDetector:
    if key not in DETECTORS:
        raise ValueError(f'Could not find detector for: "{key}". Choose from the following: %s' % ','.join(DETECTORS))
    if not detector_cache.get(key):
        detector = DETECTORS[key]
        detector_cache[key] = detector(*args, **kwargs)
    return detector_cache[key]

async def prepare(detector_key: Detector):
    detector = get_detector(detector_key)
    if isinstance(detector, OfflineDetector):
        await detector.download()
    elif isinstance(detector, SwitchDetector):
        await detector.default_detector.download()
        await detector.ctd_detector.download()

async def dispatch(detector_key: Detector, image: np.ndarray, detect_size: int, text_threshold: float, box_threshold: float, unclip_ratio: float,
                   invert: bool, gamma_correct: bool, rotate: bool, auto_rotate: bool = False, device: str = 'cpu', verbose: bool = False):
    try:
        detector = get_detector(detector_key)
        if isinstance(detector, (OfflineDetector, SwitchDetector)):
            if not hasattr(detector, 'model') or detector.model is None or isinstance(detector, SwitchDetector):
                logger.info(f"Loading {detector_key} model...")
                await detector.load(device)
                
        logger.info(f"Running detection with {detector_key}")
        return await detector.detect(image, detect_size, text_threshold, box_threshold, unclip_ratio, invert, gamma_correct, rotate, auto_rotate, verbose)
    except Exception as e:
        logger.error(f"Error in dispatch for {detector_key}: {str(e)}", exc_info=True)
        raise

async def unload(detector_key: Detector):
    detector_cache.pop(detector_key, None)