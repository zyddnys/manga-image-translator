import os
import json

import cv2
import numpy as np

from .default import DefaultDetector
from .dbnet_convnext import DBConvNextDetector
from .ctd import ComicTextDetector
from .craft import CRAFTDetector
from .paddle_rust import PaddleDetector
from .none import NoneDetector
from .yolo_manga import YoloMangaDetector
from .common import CommonDetector, OfflineDetector
from ..config import Detector
from ..utils import Quadrilateral

DETECTORS = {
    Detector.default: DefaultDetector,
    Detector.dbconvnext: DBConvNextDetector,
    Detector.ctd: ComicTextDetector,
    Detector.craft: CRAFTDetector,
    Detector.paddle: PaddleDetector,
    Detector.none: NoneDetector,
    Detector.yolomanga: YoloMangaDetector,
}
detector_cache = {}

# ── Manual region injection ──────────────────────────────────────────────────
# When the web app processes an image that has hand-drawn regions, it sets the
# env var MIT_MANUAL_REGIONS to a JSON payload BEFORE launching this subprocess
# (one image per run, so no per-image disambiguation is needed):
#
#   {"mode": "merge"|"replace",
#    "regions": [{"x": 0.62, "y": 0.30, "w": 0.31, "h": 0.06}, ...]}
#
# Coordinates are normalized 0..1 relative to the image, so they map correctly
# regardless of detection resolution / upscaling. The boxes are injected as
# empty-text Quadrilaterals (OCR fills in the text downstream) and painted onto
# the detection mask so the original text underneath is inpainted away.
#
# Behavior is strictly additive: if the env var is absent or unparseable, this
# is a no-op and dispatch() returns the detector output byte-for-byte unchanged.

_MANUAL_ENV = "MIT_MANUAL_REGIONS"


def _parse_manual_regions():
    raw = os.environ.get(_MANUAL_ENV, "").strip()
    if not raw:
        return None
    try:
        payload = json.loads(raw)
    except Exception:
        return None
    regions = payload.get("regions") if isinstance(payload, dict) else None
    if not regions:
        return None
    mode = (payload.get("mode") or "merge").lower()
    if mode not in ("merge", "replace"):
        mode = "merge"
    return mode, regions


def _apply_manual_regions(result, image: np.ndarray):
    parsed = _parse_manual_regions()
    if parsed is None:
        return result

    mode, regions = parsed
    textlines, raw_mask, mask = result
    im_h, im_w = image.shape[:2]

    manual_lines = []
    boxes_px = []
    for r in regions:
        try:
            x = float(r["x"]); y = float(r["y"])
            w = float(r["w"]); h = float(r["h"])
        except (KeyError, TypeError, ValueError):
            continue
        x0 = int(round(max(0.0, min(1.0, x)) * im_w))
        y0 = int(round(max(0.0, min(1.0, y)) * im_h))
        x1 = int(round(max(0.0, min(1.0, x + w)) * im_w))
        y1 = int(round(max(0.0, min(1.0, y + h)) * im_h))
        if x1 - x0 < 2 or y1 - y0 < 2:
            continue
        pts = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.int32)
        manual_lines.append(Quadrilateral(pts, '', 1.0))
        boxes_px.append((x0, y0, x1, y1))

    if not manual_lines:
        return result

    if mode == "replace":
        textlines = manual_lines
        # Discard auto-detected mask pixels — only the hand-drawn boxes should
        # be inpainted. Reset to a single-channel canvas sized to the image.
        if mask is not None:
            mask = np.zeros((im_h, im_w), dtype=np.uint8)
        if raw_mask is not None:
            raw_mask = np.zeros((im_h, im_w), dtype=np.uint8)
    else:
        textlines = list(textlines) + manual_lines
        # NoneDetector returns a degenerate 3-channel zero mask; downstream mask
        # refinement (cv2.connectedComponents) requires a single channel. Real
        # detectors already return a 2-D mask so this is a no-op for them.
        if raw_mask is not None and raw_mask.ndim == 3:
            raw_mask = np.zeros((im_h, im_w), dtype=np.uint8)

    # Paint the manual boxes into the mask so inpainting erases the source text.
    # For detectors that return mask=None (e.g. CTD), the mask is regenerated
    # downstream from text_regions, so the injected textlines already cover it.
    for (x0, y0, x1, y1) in boxes_px:
        if mask is not None:
            cv2.rectangle(mask, (x0, y0), (x1, y1), 255, thickness=-1)
        if raw_mask is not None:
            cv2.rectangle(raw_mask, (x0, y0), (x1, y1), 255, thickness=-1)

    return textlines, raw_mask, mask


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

async def dispatch(detector_key: Detector, image: np.ndarray, detect_size: int, text_threshold: float, box_threshold: float, unclip_ratio: float,
                   invert: bool, gamma_correct: bool, rotate: bool, auto_rotate: bool = False, device: str = 'cpu', verbose: bool = False):
    detector = get_detector(detector_key)
    if isinstance(detector, OfflineDetector):
        await detector.load(device)
    result = await detector.detect(image, detect_size, text_threshold, box_threshold, unclip_ratio, invert, gamma_correct, rotate, auto_rotate, verbose)
    return _apply_manual_regions(result, image)

async def unload(detector_key: Detector):
    detector_cache.pop(detector_key, None)
