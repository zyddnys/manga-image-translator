import os
import shutil
import numpy as np
import cv2
from paddleocr import TextDetection #采用文本检测模块而非完整的OCR模块
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
            device = 'gpu'
        else:
            self.use_gpu = False
            device = 'cpu'
        self.model=TextDetection(
            model_name='PP-OCRv5_server_det',
            model_dir=os.path.join(self.model_dir, 'PP-OCRv5_server_det_infer'),
            device=device,
        )
        global MODEL
        MODEL = self.model

    async def _unload(self):
        del self.model

    async def _infer(self, image: np.ndarray, detect_size: int, text_threshold: float, box_threshold: float,
                     unclip_ratio: float, verbose: bool = False):
        global MODEL
        
        # 使用TextDetection进行推理
        results = MODEL.predict_iter(
            image, batch_size=1,
            limit_side_len=detect_size,
            thresh=text_threshold, 
            box_thresh=box_threshold,
            unclip_ratio=unclip_ratio
        )

        textlines = []
        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        
        for result in results:
            #result.save_to_img(save_path="./output/")
            if result['dt_polys'] is not None and len(result['dt_polys']) > 0:
                # 遍历每个检测到的文本框
                for i, poly in enumerate(result['dt_polys']):

                    # 转换为整数坐标
                    poly_int = poly.astype(np.int32)

                    # 获取置信度
                    score = result['dt_scores'][i]
                    
                    textlines.append(Quadrilateral(poly_int, '', score))
                    
                    # 在mask上绘制四边形
                    cv2.fillPoly(mask, [poly_int], color=255)

        return textlines, mask, None
