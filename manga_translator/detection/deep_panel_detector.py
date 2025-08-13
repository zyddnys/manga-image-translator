#!/usr/bin/env python3
"""
Deep learning panel detector | 深度学习分镜检测器
Panel detection based on trained Faster R-CNN model | 基于训练好的Faster R-CNN模型进行分镜检测
"""

import os
import sys
import time
import numpy as np
from typing import List, Tuple, Dict, Optional
import logging

# Add panel_utils path | 添加panel_utils路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'panel_utils'))

try:
    from .panel_utils.panel_models import PanelDetectionModel, PanelDetectionConfig
    from .panel_utils.model_configs import PanelModelConfigs
    from ..utils.textblock import _sort_panels_fill as textblock_sort_panels
except ImportError:
    from panel_utils.panel_models import PanelDetectionModel, PanelDetectionConfig
    from panel_utils.model_configs import PanelModelConfigs
    from manga_translator.utils.textblock import _sort_panels_fill as textblock_sort_panels

# Set up logging | 设置日志
logger = logging.getLogger(__name__)

    
class DeepPanelDetector:    
    """
    Deep learning panel detector | 深度学习分镜检测器

    Uses trained Faster R-CNN model for panel detection, output format compatible with existing Panel system | 使用训练好的Faster R-CNN模型进行分镜检测，输出格式兼容现有的Panel系统
    """
    
    def __init__(self, model_path: str = None, config_name: str = "manga_optimized", device: str = "auto"):
        """
        Initialize deep learning panel detector | 初始化深度学习分镜检测器

        Args:
            model_path: Model weight file path, use default if None | 模型权重文件路径，如果为None则使用默认路径
            config_name: Model configuration name | 模型配置名称
            device: Device type ("auto", "cuda", "cpu") | 设备类型 ("auto", "cuda", "cpu")
        """
        self.model_path = model_path or self._get_default_model_path()
        self.config_name = config_name
        self.device = self._determine_device(device)
        
        # 模型相关
        self.model = None
        self.config = None
        self.is_initialized = False
        
        # 检测参数
        self.score_threshold = 0.5
        self.nms_threshold = 0.5
        self.max_detections = 50
        
        # Device info will be logged during actual initialization | 设备信息将在实际初始化时记录
    
    def _get_default_model_path(self) -> str:
        """Get default model path with automatic download | 获取默认模型路径并自动下载"""
        model_path = "models/panel_detection/panel_detection.pth"

        if os.path.exists(model_path):
            return model_path

        # Try to download model automatically
        try:
            self._download_model(model_path)
            return model_path
        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            raise FileNotFoundError(
                f"Model not found at {model_path} and download failed: {e}. "
                "Please download manually from: https://huggingface.co/popcion/panel_detection"
            )

    def _download_model(self, model_path: str):
        """Download model from Hugging Face | 从Hugging Face下载模型"""
        try:
            import huggingface_hub

            # Ensure directory exists
            os.makedirs(os.path.dirname(model_path), exist_ok=True)

            # Download the model file
            huggingface_hub.hf_hub_download(
                repo_id="popcion/panel_detection",
                filename="panel_detection.pth",
                local_dir=os.path.dirname(model_path),
                local_dir_use_symlinks=False
            )   
            
        except ImportError:
            raise ImportError("huggingface_hub is required. Install with: pip install huggingface_hub")
        except Exception as e:
            raise Exception(f"Download failed: {e}")
    
    def _determine_device(self, device: str) -> str:
        """Determine device to use | 确定使用的设备"""
        if device == "auto":
            try:
                import torch
                if torch.cuda.is_available():
                    return "cuda"
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    return "mps"
                else:
                    return "cpu"
            except ImportError:
                return "cpu"
        return device
    
    def initialize(self) -> bool:
        """
        Initialize model | 初始化模型

        Returns:
            bool: Whether initialization was successful | 初始化是否成功
        """
        try:
            # 检查模型文件是否存在，如果不存在尝试重新获取路径（可能触发下载）
            if not os.path.exists(self.model_path):
                logger.info(f"Model file not found at {self.model_path}, attempting to get/download...")
                try:
                    # Try to get the model path (may trigger download)
                    self.model_path = self._get_default_model_path()
                except Exception as e:
                    logger.error(f"Failed to get model path: {e}")
                    return False
            
            # 获取模型配置
            self.config = PanelModelConfigs.get_config_by_name(self.config_name)
            
            # 创建模型
            self.model = PanelDetectionModel(self.config)
            
            # 初始化模型
            self.model.initialize_model(device=self.device)
            
            # 加载权重
            success = self.model.load_model(self.model_path)
            if not success:
                logger.error("Failed to load model weights")
                return False
            
            # 验证模型
            if not self.model.validate_model():
                logger.error("Model validation failed")
                return False
            
            self.is_initialized = True
            logger.info(f"Deep learning panel detector initialized on {self.device.upper()}")

            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            return False
    
    def detect_panels(self, image: np.ndarray, rtl: bool = True) -> List[Tuple[int, int, int, int]]:
        """
        Detect panels | 检测分镜

        Args:
            image: Input image (H, W, C) BGR format | 输入图像 (H, W, C) BGR格式
            rtl: Whether reading right-to-left (for sorting) | 是否为从右到左阅读 (用于排序)

        Returns:
            List[Tuple[int, int, int, int]]: Panel list in format (x, y, w, h) | 分镜列表，格式为 (x, y, w, h)
        """
        if not self.is_initialized:
            if not self.initialize():
                logger.error("Model not initialized, falling back to empty result")
                return []
        
        try:
            # 执行推理
            results = self.model.predict(image)
            
            # 转换结果格式
            panels = []
            for result in results:
                if result['score'] >= self.score_threshold:
                    bbox = result['bbox']  # [x, y, w, h]
                    x, y, w, h = bbox
                    panels.append((int(x), int(y), int(w), int(h)))
            
            # 使用与kumiko相同的排序逻辑
            panels = self._sort_panels_fill(panels, rtl)
            

            return panels
            
        except Exception as e:
            logger.error(f"Panel detection failed: {e}")
            return []
    
    def _sort_panels_fill(self, panels: List[Tuple[int, int, int, int]], rtl: bool = True) -> List[Tuple[int, int, int, int]]:
        """
        Sort panels using improved logic with vertical stack priority | 使用改进的垂直堆叠优先排序逻辑

        Args:
            panels: Panel list [(x, y, w, h), ...] | 分镜列表 [(x, y, w, h), ...]
            rtl: Whether reading right-to-left | 是否为从右到左阅读

        Returns:
            Sorted panel list | 排序后的分镜列表
        """
        if not panels:
            return panels

        # 转换为 (x1, y1, x2, y2) 格式
        panels_xyxy = [(x, y, x + w, y + h) for x, y, w, h in panels]

        # 使用与textblock.py相同的改进排序逻辑
        sorted_xyxy = textblock_sort_panels(panels_xyxy, rtl)

        # 转换回 (x, y, w, h) 格式
        return [(x1, y1, x2 - x1, y2 - y1) for x1, y1, x2, y2 in sorted_xyxy]
    
    def get_panels_from_array(self, img_rgb: np.ndarray, rtl: bool = True) -> List[Tuple[int, int, int, int]]:
        """
        Compatible interface for existing Panel system | 兼容现有Panel系统的接口
        
        Args:
            img_rgb: RGB format image array | RGB格式图像
            rtl: Whether reading right-to-left | 是否为从右到左阅读
        
        Returns:
            Panel list in format (x, y, w, h) | 分镜列表，格式为 (x, y, w, h)
        """
        # 转换RGB到BGR（模型期望BGR格式）
        img_bgr = img_rgb[:, :, ::-1]
        
        return self.detect_panels(img_bgr, rtl)
    
    def set_detection_parameters(self, score_threshold: float = None, 
                                nms_threshold: float = None, 
                                max_detections: int = None):
        """
        Set detection parameters | 设置检测参数

        Args:
            score_threshold: Confidence threshold | 置信度阈值
            nms_threshold: NMS threshold | NMS阈值
            max_detections: Maximum number of detections | 最大检测数量
        """
        if score_threshold is not None:
            self.score_threshold = score_threshold
        if nms_threshold is not None:
            self.nms_threshold = nms_threshold
        if max_detections is not None:
            self.max_detections = max_detections
        
        logger.info(f"Detection parameters updated: score={self.score_threshold}, "
                   f"nms={self.nms_threshold}, max={self.max_detections}")
    
    def get_model_info(self) -> Dict:
        """Get model information | 获取模型信息"""
        if not self.is_initialized:
            return {
                "status": "not_initialized",
                "model_path": self.model_path,
                "device": self.device
            }
        
        model_info = self.model.get_model_info()
        model_info.update({
            "model_path": self.model_path,
            "config_name": self.config_name,
            "score_threshold": self.score_threshold,
            "nms_threshold": self.nms_threshold,
            "max_detections": self.max_detections
        })
        
        return model_info
    
    def benchmark(self, image: np.ndarray, num_runs: int = 10) -> Dict:
        """
        Performance benchmark test | 性能基准测试

        Args:
            image: Test image | 测试图像
            num_runs: Number of runs | 运行次数

        Returns:
            Performance statistics | 性能统计信息
        """
        if not self.is_initialized:
            if not self.initialize():
                return {"error": "Model not initialized"}

        times = []
        panel_counts = []
        
        # 预热
        self.detect_panels(image)
        
        # 基准测试
        for i in range(num_runs):
            start_time = time.time()
            panels = self.detect_panels(image)
            end_time = time.time()
            
            times.append(end_time - start_time)
            panel_counts.append(len(panels))
        
        return {
            "num_runs": num_runs,
            "avg_time": np.mean(times),
            "min_time": np.min(times),
            "max_time": np.max(times),
            "std_time": np.std(times),
            "avg_panels": np.mean(panel_counts),
            "fps": 1.0 / np.mean(times)
        }



