#!/usr/bin/env python3
"""
Panel detection module with unified architecture
Provides registration system and dispatch functions for panel detectors
"""

import numpy as np
from typing import List, Tuple, Optional
from .common import CommonPanelDetector, OfflinePanelDetector
from .kumiko_detector import KumikoPanelDetector
from .deep_panel_detector import DeepPanelDetector
from ..config import PanelDetector, PanelDetectorConfig

PANEL_DETECTORS = {
    PanelDetector.dl: DeepPanelDetector,
    PanelDetector.kumiko: KumikoPanelDetector,
}
panel_detector_cache = {}


def get_panel_detector(key: PanelDetector, *args, **kwargs) -> CommonPanelDetector:
    if key not in PANEL_DETECTORS:
        raise ValueError(f'Could not find panel detector for: "{key}". Choose from the following: %s' % ','.join(PANEL_DETECTORS))
    if not panel_detector_cache.get(key):
        panel_detector = PANEL_DETECTORS[key]
        panel_detector_cache[key] = panel_detector(*args, **kwargs)
    return panel_detector_cache[key]

async def prepare(detector_key: PanelDetector):
    detector = get_panel_detector(detector_key)
    if isinstance(detector, OfflinePanelDetector):
        await detector.download()

async def unload(detector_key: PanelDetector):
    panel_detector_cache.pop(detector_key, None)

async def dispatch(detector_key: PanelDetector, image: np.ndarray, rtl: bool = True, device: str = 'cpu', config: Optional[PanelDetectorConfig] = None, **kwargs) -> List[Tuple[int, int, int, int]]:
    detector = get_panel_detector(detector_key)
    if isinstance(detector, OfflinePanelDetector):
        await detector.load(device)
    config = config or PanelDetectorConfig()
    return await detector._infer(image, rtl, config=config, **kwargs)