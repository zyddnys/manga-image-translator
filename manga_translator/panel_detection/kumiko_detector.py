#!/usr/bin/env python3
"""
Kumiko panel detector implementation using traditional image processing
"""

import os
import sys
import tempfile
from typing import List, Tuple

import cv2 as cv
import numpy as np

from .common import CommonPanelDetector
from .kumiko import Page, NotAnImageException, Debug
from ..config import PanelDetectorConfig


class KumikoPanelDetector(CommonPanelDetector):
    """Kumiko-based panel detector using traditional image processing

    Does not require model loading and works directly with OpenCV
    """

    def __init__(self):
        """Initialize Kumiko panel detector"""
        super().__init__()

    async def _detect_panels(self, image: np.ndarray, rtl: bool = True, **kwargs) -> List[Tuple[int, int, int, int]]:
        """Implementation of abstract method from CommonPanelDetector"""
        return await self._infer(image, rtl, **kwargs)

    async def _infer(self, image: np.ndarray, rtl: bool = True, config: PanelDetectorConfig = None, **kwargs) -> List[Tuple[int, int, int, int]]:
        """Panel detection using Kumiko traditional image processing"""
        config = config or PanelDetectorConfig()
        Debug.debug = config.debug

        try:
            # Convert RGB to BGR for OpenCV
            image_bgr = cv.cvtColor(image, cv.COLOR_RGB2BGR)

            # Encode to memory buffer for better performance
            success, buffer = cv.imencode('.png', image_bgr)
            if not success:
                raise RuntimeError("Failed to encode image to PNG format")

            # Create temporary file (required by Kumiko API)
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                temp_filename = temp_file.name
                temp_file.write(buffer.tobytes())
                temp_file.flush()

                try:
                    # Process with Kumiko
                    page = Page(
                        filename=temp_filename,
                        numbering='rtl' if rtl else 'ltr',
                        debug=config.debug,
                        min_panel_size_ratio=config.min_panel_size_ratio,
                        panel_expansion=config.panel_expansion
                    )

                    # Extract panel coordinates
                    panels = []
                    for panel in page.panels:
                        panel_coords = (panel.x, panel.y, panel.w(), panel.h())
                        panels.append(panel_coords)

                    return panels

                finally:
                    # Clean up temporary file
                    try:
                        os.unlink(temp_filename)
                    except OSError:
                        pass

        except NotAnImageException as e:
            raise ValueError(f"Invalid image input: {e}")
        except Exception as e:
            raise RuntimeError(f"Kumiko panel detection failed: {e}")


    

