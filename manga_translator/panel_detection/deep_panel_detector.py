#!/usr/bin/env python3
"""
Deep learning panel detector based on trained Faster R-CNN model
"""

import os
import time
import numpy as np
from typing import List, Tuple, Dict
import logging

logger = logging.getLogger(__name__)

from .common import OfflinePanelDetector
from .panel_utils.panel_models import PanelDetectionModel, PanelDetectionConfig
from .panel_utils.model_configs import PanelModelConfigs
from ..config import PanelDetectorConfig


class DeepPanelDetector(OfflinePanelDetector):
    """Deep learning panel detector using Faster R-CNN"""

    _MODEL_MAPPING = {
        'model': {
            'url': 'https://huggingface.co/popcion/panel_detection/resolve/main/panel_detection_250821.pth',
            'hash': '2dc683b7fecf6ab2855df60e4022b8d426f6533c27958e66b3c934b8aa9e73d7',
            'file': 'panel_detection_250821.pth',
            # Legacy model option:
            # 'url': 'https://huggingface.co/popcion/panel_detection/resolve/main/panel_detection_250808.pth',
            # 'hash': '23a46d9901ee242819b05ad2c01bcf3c8b0a0455754418acc47248befacde22c',
            # 'file': 'panel_detection_250808.pth',
        }
    }

    def __init__(self):
        """Initialize deep learning panel detector"""
        super().__init__()
        self.config_name = "manga_optimized"
        self.model_config = None
        self.model = None

    async def _load(self, device: str, *args, **kwargs):
        """Load model into memory"""
        try:
            # Get model file name from mapping
            model_file_name = self._MODEL_MAPPING['model']['file']
            model_path = self._get_file_path(model_file_name)

            if not os.path.exists(model_path):
                logger.error(f"Model file not found at {model_path}")
                return False

            # Auto-select config based on model file
            model_file = os.path.basename(model_path)
            self.model_config = PanelModelConfigs.get_config_by_model_file(model_file)

            # Create and initialize model
            self.model = PanelDetectionModel(self.model_config)
            self.model.initialize_model(device=device)

            # Load weights
            success = self.model.load_model(model_path)
            if not success:
                logger.error("Failed to load model weights")
                return False

            logger.info(f"Panel detector loaded: {self.model_config.backbone} on {device}")
            return True

        except Exception as e:
            logger.error(f"Failed to load panel detector: {e}")
            return False

    async def _unload(self):
        """Unload model from memory"""
        if hasattr(self, 'model') and self.model is not None:
            del self.model
        if hasattr(self, 'model_config'):
            del self.model_config



    async def _infer(self, image: np.ndarray, rtl: bool = True, config: PanelDetectorConfig = None, **kwargs) -> List[Tuple[int, int, int, int]]:
        """Perform panel detection inference

        Args:
            image: Input image array (H, W, C) in RGB format
            rtl: Whether reading right-to-left
            config: Panel detector configuration

        Returns:
            List of panels in format (x, y, w, h)
        """
        if self.model is None:
            logger.error("Model not loaded")
            return []

        try:
            # Convert RGB to BGR for model
            image_bgr = image[:, :, ::-1]

            # Execute inference (filtering handled by detectron2)
            results = self.model.predict(image_bgr)

            # Convert to expected format
            panels = []
            for result in results:
                bbox = result['bbox']  # [x, y, w, h]
                x, y, w, h = bbox
                panels.append((int(x), int(y), int(w), int(h)))

            return panels

        except Exception as e:
            logger.error(f"Panel detection failed: {e}")
            return []



