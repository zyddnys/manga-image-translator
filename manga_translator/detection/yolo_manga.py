"""
YOLOv8-Manga Detector with SFX/artistic text support.

Model: keremberke/yolov8m-manga-text-detection (HuggingFace)
Weight URL: https://huggingface.co/keremberke/yolov8m-manga-text-detection/resolve/main/best.pt

Integrates into MIT pipeline as Detector.yolomanga without modifying any
existing detector or the core text-reflow pipeline.
"""

from __future__ import annotations

import asyncio
import os
import shutil
from typing import List, Tuple

import cv2
import numpy as np
import torch

from .common import OfflineDetector
from ..utils import Quadrilateral


# ---------------------------------------------------------------------------
# Shape / SFX filter helpers
# ---------------------------------------------------------------------------

def _aspect_ratio_ok(x1: int, y1: int, x2: int, y2: int,
                     max_ratio: float = 8.0) -> bool:
    """Reject boxes that are absurdly wide/tall (pure decorative lines)."""
    w, h = max(x2 - x1, 1), max(y2 - y1, 1)
    ratio = max(w, h) / min(w, h)
    return ratio <= max_ratio


def _min_area_ok(x1: int, y1: int, x2: int, y2: int,
                 min_pixels: int = 64) -> bool:
    """Reject tiny detections (noise)."""
    return (x2 - x1) * (y2 - y1) >= min_pixels


def _sfx_heuristic(crop: np.ndarray) -> bool:
    """
    Return True for SFX-like crops:
    - High edge density (jagged / artistic lettering)
    - OR very dark/high-contrast region (shaded speech bubble interior)
    Uses Canny edge density as a proxy for non-rectangular text shapes.
    """
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY) if crop.ndim == 3 else crop
    edges = cv2.Canny(gray, 50, 150)
    edge_density = edges.sum() / (gray.size + 1e-6)
    return float(edge_density) > 0.04  # tuneable threshold


# ---------------------------------------------------------------------------
# Detector class
# ---------------------------------------------------------------------------

class YoloMangaDetector(OfflineDetector):
    """
    Drop-in MIT detector using YOLOv8m fine-tuned on manga text.

    _MODEL_MAPPING follows MIT's ModelWrapper convention so that
    `prepare()` / download logic is handled automatically.
    """

    _MODEL_MAPPING = {
        'model': {
            'url': (
                'https://huggingface.co/keremberke/yolov8m-manga-text-detection'
                '/resolve/main/best.pt'
            ),
            'hash': None,          # set after first download verification
            'file': 'yolov8m_manga.pt',
        }
    }

    # SFX-confidence boost: if YOLO conf < this but sfx_heuristic passes, keep
    SFX_CONF_BOOST_THRESHOLD = 0.25
    # Normal conf threshold (overridden by MIT's box_threshold arg)
    DEFAULT_CONF = 0.35

    def __init__(self, *args, **kwargs):
        os.makedirs(self.model_dir, exist_ok=True)
        super().__init__(*args, **kwargs)

    async def _load(self, device: str):
        from ultralytics import YOLO
        weight_path = self._get_file_path('yolov8m_manga.pt')
        self.model = YOLO(weight_path)
        self.device = device
        # Move to GPU if available
        if device.startswith('cuda'):
            self.model.to(device)
        self.logger.info(f'YoloMangaDetector loaded on {device}')

    async def _unload(self):
        del self.model

    # ------------------------------------------------------------------
    # Core _detect — must return (textlines, raw_mask, mask)
    # ------------------------------------------------------------------

    async def _detect(
        self,
        image: np.ndarray,
        detect_size: int,
        text_threshold: float,
        box_threshold: float,
        unclip_ratio: float,
        verbose: bool = False,
    ) -> Tuple[List[Quadrilateral], np.ndarray, np.ndarray]:

        img_h, img_w = image.shape[:2]

        # Resize to detect_size keeping aspect ratio
        scale = detect_size / max(img_h, img_w)
        scaled_w = int(img_w * scale)
        scaled_h = int(img_h * scale)
        img_resized = cv2.resize(image, (scaled_w, scaled_h),
                                 interpolation=cv2.INTER_LINEAR)

        # Run YOLO (returns list of Results)
        conf_threshold = max(box_threshold, self.DEFAULT_CONF)
        results = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.model.predict(
                source=img_resized,
                conf=self.SFX_CONF_BOOST_THRESHOLD,  # low floor; we filter below
                iou=0.45,
                imgsz=detect_size,
                device=self.device,
                verbose=False,
            )
        )

        textlines: List[Quadrilateral] = []
        mask = np.zeros((img_h, img_w), dtype=np.uint8)

        if not results or results[0].boxes is None:
            return textlines, mask.copy(), mask

        boxes = results[0].boxes
        for box in boxes:
            conf = float(box.conf[0])
            x1, y1, x2, y2 = [int(v / scale) for v in box.xyxy[0].tolist()]

            # Clamp to image bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img_w, x2), min(img_h, y2)

            if not _min_area_ok(x1, y1, x2, y2):
                continue
            if not _aspect_ratio_ok(x1, y1, x2, y2):
                continue

            # SFX boost: keep low-confidence detection if heuristic agrees
            if conf < conf_threshold:
                crop = image[y1:y2, x1:x2]
                if not _sfx_heuristic(crop):
                    continue

            # Build Quadrilateral (MIT's standard box format)
            pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                           dtype=np.float32)
            quad = Quadrilateral(pts, '', conf)
            textlines.append(quad)

            # Draw on mask
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

        raw_mask = mask.copy()

        if verbose:
            self.logger.debug(
                f'YoloManga: {len(textlines)} boxes detected '
                f'(conf>={conf_threshold:.2f} + SFX heuristic)'
            )

        return textlines, raw_mask, mask
