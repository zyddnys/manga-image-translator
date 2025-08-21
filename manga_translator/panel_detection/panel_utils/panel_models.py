#!/usr/bin/env python3
"""
Panel detection model architecture based on Faster R-CNN
"""

import os
import torch
from typing import Dict, List, Any
import numpy as np
from dataclasses import dataclass

try:
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor
    from detectron2.model_zoo import model_zoo
    DETECTRON2_AVAILABLE = True
except ImportError:
    DETECTRON2_AVAILABLE = False


@dataclass
class PanelDetectionConfig:
    """Panel detection model configuration"""

    # Model architecture - high precision model default
    model_name: str = "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"
    backbone: str = "ResNet101"
    num_classes: int = 1  # Panel class only

    # Input configuration - high resolution
    min_size_test: int = 1000
    max_size_test: int = 1600

    # Anchor configuration
    anchor_sizes: List[List[int]] = None  # Set in __post_init__
    anchor_aspect_ratios: List[List[float]] = None

    # RPN configuration - high precision
    rpn_pre_nms_topk_test: int = 3000
    rpn_post_nms_topk_test: int = 1500
    rpn_nms_thresh: float = 0.6

    # ROI configuration - kept for compatibility, not used in inference
    roi_batch_size_per_image: int = 256
    roi_positive_fraction: float = 0.25
    roi_iou_thresholds: List[float] = None

    # NMS configuration - high precision
    nms_thresh_test: float = 0.4
    score_thresh_test: float = 0.6
    detections_per_img: int = 50

    def __post_init__(self):
        """Post-initialization setup"""
        if self.anchor_sizes is None:
            # High precision anchor sizes
            self.anchor_sizes = [[64], [128], [256], [512], [768]]

        if self.anchor_aspect_ratios is None:
            # High precision aspect ratios
            self.anchor_aspect_ratios = [[0.2, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0]] * 5

        if self.roi_iou_thresholds is None:
            self.roi_iou_thresholds = [0.6]


class PanelDetectionModel:
    """Panel detection model wrapper class"""
    
    def __init__(self, config: PanelDetectionConfig):
        self.config = config
        self.model = None
        self.predictor = None
        self.device = None
        
        if not DETECTRON2_AVAILABLE:
            raise ImportError(
                "detectron2 is required for panel detection. "
                "Please install it"
            )
    
    def create_config(self) -> Any:
        """Create detectron2 configuration"""
        cfg = get_cfg()

        # Base model configuration
        cfg.merge_from_file(model_zoo.get_config_file(self.config.model_name))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(self.config.model_name)

        # Model configuration
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = self.config.num_classes
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.config.score_thresh_test
        cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = self.config.nms_thresh_test

        # Anchor configuration
        cfg.MODEL.ANCHOR_GENERATOR.SIZES = self.config.anchor_sizes
        cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = self.config.anchor_aspect_ratios

        # RPN configuration
        cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = self.config.rpn_pre_nms_topk_test
        cfg.MODEL.RPN.POST_NMS_TOPK_TEST = self.config.rpn_post_nms_topk_test
        cfg.MODEL.RPN.NMS_THRESH = self.config.rpn_nms_thresh

        # ROI configuration - not used in inference, kept for training compatibility
        # cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = self.config.roi_batch_size_per_image
        # cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION = self.config.roi_positive_fraction
        # cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS = self.config.roi_iou_thresholds

        # Input configuration
        cfg.INPUT.MIN_SIZE_TEST = self.config.min_size_test
        cfg.INPUT.MAX_SIZE_TEST = self.config.max_size_test

        # Test configuration
        cfg.TEST.DETECTIONS_PER_IMAGE = self.config.detections_per_img

        return cfg
    
    def initialize_model(self, device: str = "cpu") -> None:
        """Initialize model"""
        self.device = device
        cfg = self.create_config()

        # Set device (already validated in main program)
        cfg.MODEL.DEVICE = device

        # Create predictor
        self.predictor = DefaultPredictor(cfg)
        self.model = self.predictor.model

    def load_model(self, model_path: str) -> bool:
        """Load trained model weights"""
        try:
            if not os.path.exists(model_path):
                print(f"Model file not found: {model_path}")
                return False

            if self.predictor is None:
                print("Model must be initialized before loading weights")
                return False

            # Load weights to model, supports CUDA, MPS and CPU
            if self.device == "cuda" and torch.cuda.is_available():
                load_device = "cuda"
            elif self.device == "mps" and torch.backends.mps.is_available():
                load_device = "mps"
            else:
                load_device = "cpu"
            checkpoint = torch.load(model_path, map_location=load_device)

            # Handle different weight formats
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint

            # Load weights
            self.model.load_state_dict(state_dict, strict=False)
            return True

        except Exception as e:
            print(f"Failed to load model weights: {e}")
            return False

    def predict(self, image: np.ndarray) -> List[Dict]:
        """Execute inference"""
        if self.predictor is None:
            raise RuntimeError("Model must be initialized before prediction")
        
        # Execute inference
        outputs = self.predictor(image)
        
        # Extract results
        instances = outputs["instances"].to("cpu")
        
        results = []
        if len(instances) > 0:
            boxes = instances.pred_boxes.tensor.numpy()
            scores = instances.scores.numpy()
            classes = instances.pred_classes.numpy()
            
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes[i]
                result = {
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],  # [x, y, w, h]
                    "score": float(scores[i]),
                    "class": int(classes[i]),
                    "class_name": "panel"
                }
                results.append(result)
        
        return results