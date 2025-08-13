#!/usr/bin/env python3
"""
深度学习Panel检测接口
将深度学习检测器输出转换为Panel对象格式，保持与现有接口的完全兼容
"""

import numpy as np
from typing import List, Tuple
import logging

# Import deep panel detector | 导入深度学习分镜检测器
from manga_translator.detection.deep_panel_detector import DeepPanelDetector

# 设置日志
logger = logging.getLogger(__name__)

class DLPanelInterface:
    """
    Deep learning panel detection interface | 深度学习Panel检测接口

    Responsible for calling deep learning detector and converting results to Kumiko-compatible format
    负责调用深度学习检测器并将结果转换为与Kumiko兼容的格式
    """

    # Class constants | 类常量
    MIN_PANEL_SIZE = 10  # Minimum panel size | 最小分镜尺寸

    def __init__(self, use_gpu: bool = False):
        """
        Initialize deep learning panel interface | 初始化深度学习Panel接口

        Args:
            use_gpu: Whether to use GPU (default False), corresponds to --use-gpu parameter
                    是否使用GPU (默认False)，对应manga-image-translator的--use-gpu参数
        """
        self.use_gpu = use_gpu
        self.detector = None
        self.is_initialized = False
        self._initialize_detector()
    
    def _initialize_detector(self):
        """Initialize deep learning detector | 初始化深度学习检测器"""
        try:
            # Create detector with appropriate device setting | 使用适当的设备设置创建检测器
            device = "auto" if self.use_gpu else "cpu"
            self.detector = DeepPanelDetector(device=device)
            self.is_initialized = self.detector.initialize()

            if not self.is_initialized:
                logger.error("Deep learning panel detector initialization failed")

        except Exception as e:
            logger.error(f"Failed to load deep learning panel detector: {e}")
            self.is_initialized = False
    
    def detect_panels(self, img_rgb: np.ndarray, rtl: bool = True) -> List[Tuple[int, int, int, int]]:
        """
        Detect panels | 检测分镜

        Args:
            img_rgb: RGB format image array | RGB格式的图像数组
            rtl: Whether reading right-to-left | 是否为从右到左阅读

        Returns:
            List[Tuple[int, int, int, int]]: Panel list in format (x, y, w, h) | 分镜列表，格式为 (x, y, w, h)
        """
        if not self.is_initialized:
            raise RuntimeError("Deep learning detector not properly initialized")

        try:
            # Call deep learning detector | 调用深度学习检测器
            panels = self.detector.get_panels_from_array(img_rgb, rtl)

            # Convert format and validate | 转换格式并验证
            validated_panels = self._validate_and_convert_panels(panels, img_rgb.shape)

            return validated_panels

        except Exception as e:
            logger.error(f"Deep learning panel detection failed: {e}")
            raise
    
    def _validate_and_convert_panels(self, panels: List[Tuple[int, int, int, int]], 
                                   image_shape: Tuple[int, int, int]) -> List[Tuple[int, int, int, int]]:
        """
        Validate and convert panel format | 验证和转换分镜格式

        Args:
            panels: Detected panel list | 检测到的分镜列表
            image_shape: Image shape (H, W, C) | 图像形状 (H, W, C)

        Returns:
            Validated panel list | 验证后的分镜列表
        """
        if not panels:
            return []
        
        height, width = image_shape[:2]
        validated_panels = []
        
        for panel in panels:
            try:
                # Ensure 4-tuple format | 确保是4元组格式
                if len(panel) != 4:
                    logger.warning(f"Skipping invalid panel format: {panel}")
                    continue

                x, y, w, h = panel

                # Convert to integers | 转换为整数
                x, y, w, h = int(x), int(y), int(w), int(h)

                # Boundary check and correction | 边界检查和修正
                x = max(0, min(x, width - 1))
                y = max(0, min(y, height - 1))
                w = max(1, min(w, width - x))
                h = max(1, min(h, height - y))

                # Minimum size check | 最小尺寸检查
                if w >= self.MIN_PANEL_SIZE and h >= self.MIN_PANEL_SIZE:
                    validated_panels.append((x, y, w, h))
                else:
                    logger.debug(f"Skipping too small panel: ({x}, {y}, {w}, {h})")

            except (ValueError, TypeError) as e:
                logger.warning(f"Skipping invalid panel data {panel}: {e}")
                continue
        
        return validated_panels
    
    def get_detector_info(self) -> dict:
        """Get detector information | 获取检测器信息"""
        if not self.is_initialized:
            return {
                "status": "not_initialized",
                "error": "Detector not initialized"
            }

        try:
            return self.detector.get_model_info()
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def set_detection_parameters(self, **kwargs):
        """Set detection parameters | 设置检测参数"""
        if self.is_initialized and self.detector:
            try:
                self.detector.set_detection_parameters(**kwargs)
                logger.info(f"Detection parameters updated: {kwargs}")
            except Exception as e:
                logger.error(f"Failed to set detection parameters: {e}")
        else:
            logger.warning("Detector not initialized, cannot set parameters")


# Global interface instance (lazy initialization) | 全局接口实例（延迟初始化）
_global_interface = None
_global_use_gpu = False  # Default not to use GPU | 默认不使用GPU

def get_global_dl_interface(use_gpu: bool = None) -> DLPanelInterface:
    """
    Get global deep learning panel interface instance | 获取全局深度学习Panel接口实例

    Args:
        use_gpu: Whether to use GPU, if None use global setting | 是否使用GPU，如果为None则使用全局设置
    """
    global _global_interface, _global_use_gpu

    # If use_gpu parameter is specified, update global setting | 如果指定了use_gpu参数，更新全局设置
    if use_gpu is not None:
        if _global_use_gpu != use_gpu:
            _global_use_gpu = use_gpu
            # If setting changed, recreate interface | 如果设置改变，重新创建接口
            if _global_interface is not None:
                logger.debug(f"GPU setting changed ({not use_gpu} -> {use_gpu}), recreating interface")
                _global_interface = None

    if _global_interface is None:
        _global_interface = DLPanelInterface(use_gpu=_global_use_gpu)
    return _global_interface