from __future__ import annotations

from typing import List

import cv2
import numpy as np
from pathlib import Path
import os

from .common import OfflineDetector
from ..utils import Quadrilateral
from ultralytics import YOLO


class YoloDetector(OfflineDetector):
    """
    Text detector backed by a fine-tuned YOLO26l_animetext model.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = None
        self.device = "cpu"
        self.min_det_size = int(os.environ.get("yolo_min_det_size", 640))
        self.max_det_size = int(os.environ.get("yolo_max_det_size", 1280))
        self.textline_padding_px = int(os.environ.get("yolo_textline_padding_px", 8))

    async def _load(self, device: str) -> None:
        self.device = device
        self.model = YOLO(
            os.path.join(os.path.dirname(__file__), "yolo_models", "yolo12l_animetext_finetuned_768_v2.1.pt")
        )
        self.model.to(self.device)

    async def _unload(self) -> None:
        self.model = None

    async def _infer(
        self,
        image: np.ndarray,
        detect_size: int,
        text_threshold: float,
        box_threshold: float,
        unclip_ratio: float,
        verbose: bool = False,
    ):
        height, width = image.shape[:2]
        raw_mask = np.zeros((height, width), dtype=np.uint8)
        image_size = (
            self.max_det_size
            if width > 1.5 * self.max_det_size or height > 1.5 * self.max_det_size
            else self.min_det_size
        )
        conf_threshold = float(np.clip(box_threshold, 0.0, 1.0))
        results = self.model.predict(
            source=image,
            imgsz=image_size,
            conf=conf_threshold,
            verbose=verbose,
            device=self.device,
        )

        if not results:
            return [], raw_mask, None

        result = results[0]
        if result.boxes is None or result.boxes.xyxy is None:
            return [], raw_mask, None

        boxes = result.boxes.xyxy.detach().cpu().numpy()
        scores = result.boxes.conf.detach().cpu().numpy()

        textlines: List[Quadrilateral] = []
        for box, score in zip(boxes, scores):
            x1_f, y1_f, x2_f, y2_f = box.astype(np.float32)
            x1_f -= self.textline_padding_px
            y1_f -= self.textline_padding_px
            x2_f += self.textline_padding_px
            y2_f += self.textline_padding_px

            x1, y1, x2, y2 = np.round([x1_f, y1_f, x2_f, y2_f]).astype(np.int32)
            x1 = int(np.clip(x1, 0, width - 1))
            y1 = int(np.clip(y1, 0, height - 1))
            x2 = int(np.clip(x2, 0, width - 1))
            y2 = int(np.clip(y2, 0, height - 1))
            if x2 <= x1 or y2 <= y1:
                continue

            pts = np.array(
                [
                    [x1, y1],
                    [x2, y1],
                    [x2, y2],
                    [x1, y2],
                ],
                dtype=np.int32,
            )

            quad = Quadrilateral(pts, "", float(score))
            if quad.area <= 16:
                continue

            textlines.append(quad)
            cv2.fillPoly(raw_mask, [pts], 255)

        return textlines, raw_mask, None
