#!/usr/bin/env python3
"""
分镜检测模型架构定义
基于Faster R-CNN实现单类别分镜检测
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
    """分镜检测模型配置"""
    
    # 模型架构配置
    model_name: str = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
    backbone: str = "ResNet50"
    num_classes: int = 1  # 只检测panel类别
    
    # 输入配置
    min_size_test: int = 800
    max_size_test: int = 1333
    
    # Anchor配置
    anchor_sizes: List[List[int]] = None  # 将在__post_init__中设置
    anchor_aspect_ratios: List[List[float]] = None
    
    # RPN配置
    rpn_pre_nms_topk_test: int = 1000
    rpn_post_nms_topk_test: int = 1000
    rpn_nms_thresh: float = 0.7
    
    # ROI配置
    roi_batch_size_per_image: int = 512
    roi_positive_fraction: float = 0.25
    roi_iou_thresholds: List[float] = None
    
    # NMS配置
    nms_thresh_test: float = 0.5
    score_thresh_test: float = 0.5
    detections_per_img: int = 100
    

    
    def __post_init__(self):
        """初始化后处理"""
        if self.anchor_sizes is None:
            # 基于分镜框尺寸分布设计anchor
            self.anchor_sizes = [[32], [64], [128], [256], [512]]
        
        if self.anchor_aspect_ratios is None:
            # 基于分镜框长宽比分布设计
            self.anchor_aspect_ratios = [[0.5, 1.0, 2.0]] * 5
        
        if self.roi_iou_thresholds is None:
            self.roi_iou_thresholds = [0.5]


class PanelDetectionModel:
    """分镜检测模型封装类"""
    
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
        """创建detectron2配置"""
        cfg = get_cfg()
        
        # 基础模型配置
        cfg.merge_from_file(model_zoo.get_config_file(self.config.model_name))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(self.config.model_name)

        # 模型配置
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = self.config.num_classes
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.config.score_thresh_test
        cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = self.config.nms_thresh_test
        
        # Anchor配置
        cfg.MODEL.ANCHOR_GENERATOR.SIZES = self.config.anchor_sizes
        cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = self.config.anchor_aspect_ratios
        
        # RPN配置
        cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = self.config.rpn_pre_nms_topk_test
        cfg.MODEL.RPN.POST_NMS_TOPK_TEST = self.config.rpn_post_nms_topk_test
        cfg.MODEL.RPN.NMS_THRESH = self.config.rpn_nms_thresh
        
        # ROI配置
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = self.config.roi_batch_size_per_image
        cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION = self.config.roi_positive_fraction
        cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS = self.config.roi_iou_thresholds
        
        # 输入配置
        cfg.INPUT.MIN_SIZE_TEST = self.config.min_size_test
        cfg.INPUT.MAX_SIZE_TEST = self.config.max_size_test

        # 测试配置
        cfg.TEST.DETECTIONS_PER_IMAGE = self.config.detections_per_img
        
        return cfg
    
    def initialize_model(self, device: str = "cpu") -> None:
        """初始化模型"""
        self.device = device
        cfg = self.create_config()

        # 设置设备
        if device == "cuda" and torch.cuda.is_available():
            cfg.MODEL.DEVICE = "cuda"
        elif device == "mps" and torch.backends.mps.is_available():
            cfg.MODEL.DEVICE = "mps"
        else:
            cfg.MODEL.DEVICE = "cpu"

        # 创建预测器
        self.predictor = DefaultPredictor(cfg)
        self.model = self.predictor.model

    def load_model(self, model_path: str) -> bool:
        """加载训练好的模型权重"""
        try:
            if not os.path.exists(model_path):
                print(f"Model file not found: {model_path}")
                return False

            if self.predictor is None:
                print("Model must be initialized before loading weights")
                return False

            # 加载权重到模型
            # 始终以CPU或可用的CUDA加载checkpoint，避免不被支持的map_location字符串
            load_device = "cuda" if (self.device == "cuda" and torch.cuda.is_available()) else "cpu"
            checkpoint = torch.load(model_path, map_location=load_device)

            # 处理不同的权重格式
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint

            # 加载权重
            self.model.load_state_dict(state_dict, strict=False)
            return True

        except Exception as e:
            print(f"Failed to load model weights: {e}")
            return False

    def predict(self, image: np.ndarray) -> List[Dict]:
        """执行推理"""
        if self.predictor is None:
            raise RuntimeError("Model must be initialized before prediction")
        
        # 执行推理
        outputs = self.predictor(image)
        
        # 提取结果
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

    def validate_model(self) -> bool:
        """验证模型是否正常工作"""
        return hasattr(self, 'predictor') and self.predictor is not None

    def get_model_info(self) -> Dict:
        """获取模型信息"""
        if self.model is None:
            return {"status": "not_initialized"}
        
        # 计算模型参数数量
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            "status": "initialized",
            "device": self.device,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_mb": total_params * 4 / (1024 * 1024),  # 假设float32
            "config": self.config
        }


def create_default_panel_model(device: str = "cpu") -> PanelDetectionModel:
    """创建默认配置的分镜检测模型"""
    config = PanelDetectionConfig()
    model = PanelDetectionModel(config)
    model.initialize_model(device)
    return model
