#!/usr/bin/env python3
"""
分镜检测模型配置预设
"""

from typing import Dict

try:
    from .panel_models import PanelDetectionConfig
except ImportError:
    from panel_models import PanelDetectionConfig


class PanelModelConfigs:
    """分镜检测模型配置预设"""

    @staticmethod
    def get_manga_optimized_config() -> PanelDetectionConfig:
        """漫画优化配置 - 基于训练案例优化的推理配置"""
        return PanelDetectionConfig(
            model_name="COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml",
            backbone="ResNet50",

            # 推理配置
            score_thresh_test=0.5,
            nms_thresh_test=0.5,
            detections_per_img=50,

            # 输入配置
            min_size_test=400,
            max_size_test=600,

            # Anchor配置（匹配FPN的5个特征层）
            anchor_sizes=[[32], [64], [128], [256], [512]],
            anchor_aspect_ratios=[[0.5, 1.0, 2.0]] * 5,
        )

    @staticmethod
    def get_config_by_name(config_name: str) -> PanelDetectionConfig:
        """根据名称获取配置"""
        # 目前只支持一种配置，所有名称都返回相同配置
        return PanelModelConfigs.get_manga_optimized_config()

