import numpy as np
from abc import abstractmethod
from typing import List, Tuple

from ..utils.inference import InfererModule, ModelWrapper
from ..config import PanelDetectorConfig


class CommonPanelDetector(InfererModule):
    """Base class for all panel detectors"""
    
    def __init__(self):
        super().__init__()
    
    async def detect_panels(self, image: np.ndarray, rtl: bool = True, **kwargs) -> List[Tuple[int, int, int, int]]:
        return await self._detect_panels(image, rtl, **kwargs)
    
    @abstractmethod
    async def _detect_panels(self, image: np.ndarray, rtl: bool = True, **kwargs) -> List[Tuple[int, int, int, int]]:
        pass


class OfflinePanelDetector(CommonPanelDetector, ModelWrapper):
    """Base class for panel detectors that require model loading"""
    
    _MODEL_SUB_DIR = 'panel_detection'
    
    async def _detect_panels(self, *args, **kwargs):
        return await self.infer(*args, **kwargs)
    
    @abstractmethod
    async def _load(self, device: str, *args, **kwargs):
        pass
    
    @abstractmethod
    async def _unload(self):
        pass
    
    @abstractmethod
    async def _infer(self, image: np.ndarray, rtl: bool = True, **kwargs) -> List[Tuple[int, int, int, int]]:
        pass
