#!/usr/bin/env python3
"""
Panel detection model configuration presets
"""

from typing import Dict

from .panel_models import PanelDetectionConfig


class PanelModelConfigs:
    """Panel detection model configuration presets"""

    @staticmethod
    def get_manga_optimized_config() -> PanelDetectionConfig:
        """Manga optimized configuration for legacy models"""
        return PanelDetectionConfig(
            # Model identification
            model_name="COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml",
            backbone="ResNet50",

            # Inference configuration
            score_thresh_test=0.5,
            nms_thresh_test=0.5,
            detections_per_img=50,

            # Input configuration
            min_size_test=400,
            max_size_test=600,

            # Anchor configuration (5 FPN feature layers)
            anchor_sizes=[[32], [64], [128], [256], [512]],
            anchor_aspect_ratios=[[0.5, 1.0, 2.0]] * 5,

            # RPN configuration
            rpn_pre_nms_topk_test=1000,
            rpn_post_nms_topk_test=1000,
            rpn_nms_thresh=0.7,
        )

    @staticmethod
    def get_high_precision_config() -> PanelDetectionConfig:
        """High precision configuration for new trained models"""
        return PanelDetectionConfig(
            # Model identification
            model_name="COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml",
            backbone="ResNet101",

            # Inference configuration
            score_thresh_test=0.6,
            nms_thresh_test=0.4,
            detections_per_img=50,

            # Input configuration
            min_size_test=1000,
            max_size_test=1600,

            # Anchor configuration (5 FPN feature layers)
            anchor_sizes=[[64], [128], [256], [512], [768]],
            anchor_aspect_ratios=[[0.2, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0]] * 5,

            # RPN configuration
            rpn_pre_nms_topk_test=3000,
            rpn_post_nms_topk_test=1500,
            rpn_nms_thresh=0.6,
        )

    @staticmethod
    def get_config_by_model_file(model_file: str) -> PanelDetectionConfig:
        """Auto-select configuration based on model file"""
        if "250821" in model_file or "high_precision" in model_file:
            # New high precision model
            return PanelModelConfigs.get_high_precision_config()
        else:
            # Legacy model or default
            return PanelModelConfigs.get_manga_optimized_config()

    @staticmethod
    def get_config_by_name(config_name: str) -> PanelDetectionConfig:
        """Get configuration by name"""
        if config_name == "high_precision":
            return PanelModelConfigs.get_high_precision_config()
        else:
            return PanelModelConfigs.get_manga_optimized_config()

