import os
import shutil
import numpy as np
import cv2
from paddleocr import PaddleOCR  # Use stable public API
from typing import List, Tuple

from .common import OfflineDetector
from ..utils import TextBlock, Quadrilateral
from ..utils.inference import ModelWrapper

MODEL = None

class PaddleDetector(OfflineDetector, ModelWrapper):
    _MODEL_MAPPING = {
        'det': {
            'url': 'https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv5_server_det_infer.tar',
            'hash': '22a33e0ba6a21425ea4192da03bf4395c9a0c67902bd924b7328fc859073045d',
            'archive': {
                'PP-OCRv5_server_det_infer/inference.pdiparams': 'PP-OCRv5_server_det_infer/',
                'PP-OCRv5_server_det_infer/inference.json': 'PP-OCRv5_server_det_infer/',
                'PP-OCRv5_server_det_infer/inference.yml': 'PP-OCRv5_server_det_infer/',
            },
        }
    }

    def __init__(self, *args, **kwargs):
        ModelWrapper.__init__(self)
        super().__init__(*args, **kwargs)

    async def _load(self, device: str):
        if device in ['cuda', 'mps']:
            self.use_gpu = True
            use_gpu = True
        else:
            self.use_gpu = False
            use_gpu = False

        # Use PaddleOCR in detection-only mode
        self.model = PaddleOCR(
            det=True,
            rec=False,
            use_angle_cls=False,
            lang='en',
            use_gpu=use_gpu,
            det_model_dir=os.path.join(self.model_dir, 'PP-OCRv5_server_det_infer'),
        )
        global MODEL
        MODEL = self.model

    async def _unload(self):
        del self.model

    async def _infer(self, image: np.ndarray, detect_size: int, text_threshold: float, box_threshold: float,
                     unclip_ratio: float, verbose: bool = False):
        global MODEL

        # Run detection
        results = MODEL.ocr(image, cls=False)

        textlines = []
        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

        for line in results:
            for box_info in line:
                box, _, score = box_info  # box: 4 points, score is float

                if score < text_threshold:
                    continue

                poly_int = np.array(box).astype(np.int32)
                textlines.append(Quadrilateral(poly_int, '', score))
                cv2.fillPoly(mask, [poly_int], color=255)

        return textlines, mask, None
