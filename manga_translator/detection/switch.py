import os 
import numpy as np
from typing import List, Tuple
import logging
import cv2

from .common import CommonDetector
from .default import DefaultDetector
from .ctd import ComicTextDetector
from ..utils import Quadrilateral

logger = logging.getLogger('manga_translator')

class SwitchDetector(CommonDetector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.default_detector = DefaultDetector(*args, **kwargs)
        self.ctd_detector = ComicTextDetector(*args, **kwargs)

        default_threshold = 0.550
        env_threshold_str = os.environ.get("switch_detection_aspect_ratio_threshold")
        if env_threshold_str:
            self.aspect_ratio_threshold = float(env_threshold_str)
        else:
            self.aspect_ratio_threshold = default_threshold

        logger.info(f"SwitchDetector initialized with aspect_ratio_threshold: {self.aspect_ratio_threshold}")

    async def load(self, device: str):
        """Load models for both detectors."""
        logger.info(f"Loading models on device: {device}")
        await self.default_detector.load(device)
        await self.ctd_detector.load(device)
        logger.info("Both detector models loaded successfully")

    async def _detect(self, image: np.ndarray, detect_size: int, text_threshold: float, box_threshold: float,
                    unclip_ratio: float, verbose: bool = False) -> Tuple[List[Quadrilateral], np.ndarray, np.ndarray]:
        """Detection that selects detector based on image aspect ratio."""
        logger.info(f"Starting aspect ratio-based detection with image shape: {image.shape}")
        
        orig_h, orig_w = image.shape[:2]
        aspect_ratio = orig_w / orig_h

        print(f"Aspect Ratio threashold: {self.aspect_ratio_threshold}")

        if aspect_ratio < self.aspect_ratio_threshold:
            logger.info(f"Aspect ratio: {aspect_ratio:.3f} - Using CTD detector for portrait-oriented image")
            selected_detector = self.ctd_detector
        else:
            logger.info(f"Aspect ratio: {aspect_ratio:.3f} - Using default detector for manga square style")
            selected_detector = self.default_detector
        
        scale = detect_size / max(orig_h, orig_w)
        new_h, new_w = int(orig_h * scale), int(orig_w * scale)
        resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        detection_params = {
            'detect_size': detect_size,
            'text_threshold': text_threshold,
            'box_threshold': box_threshold,
            'unclip_ratio': unclip_ratio,
            'invert': False,
            'gamma_correct': False,
            'rotate': False,
            'auto_rotate': False,
            'verbose': verbose
        }
        
        try:
            textlines, raw_mask, mask = await selected_detector.detect(resized_image, **detection_params)
            
            if new_w != orig_w or new_h != orig_h:
                textlines = self._scale_quadrilaterals(textlines, new_w, new_h, orig_w, orig_h)
                
                if raw_mask is not None:
                    raw_mask = cv2.resize(raw_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
                if mask is not None:
                    mask = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            
            logger.info(f"Detection completed with {len(textlines)} textlines")
            return textlines, raw_mask, mask
            
        except Exception as e:
            logger.error(f"Error in detection with {selected_detector.__class__.__name__}: {str(e)}", exc_info=True)
            try:
                fallback_detector = self.default_detector if selected_detector == self.ctd_detector else self.ctd_detector
                fallback_name = "DefaultDetector" if fallback_detector == self.default_detector else "CTD"
                logger.info(f"Attempting fallback to {fallback_name}")
                
                recovery_params = detection_params.copy()
                recovery_params['gamma_correct'] = True
                
                textlines, raw_mask, mask = await fallback_detector.detect(resized_image, **recovery_params)
                
                if new_w != orig_w or new_h != orig_h:
                    textlines = self._scale_quadrilaterals(textlines, new_w, new_h, orig_w, orig_h)
                    
                    if raw_mask is not None:
                        raw_mask = cv2.resize(raw_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
                    if mask is not None:
                        mask = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
                
                logger.info(f"Fallback detection completed with {len(textlines)} textlines")
                return textlines, raw_mask, mask
            except Exception as fallback_e:
                logger.error(f"Fallback detection failed: {str(fallback_e)}", exc_info=True)
                raise RuntimeError(f"Both detectors failed. Original error: {str(e)}, Fallback error: {str(fallback_e)}")
    
    def _scale_quadrilaterals(self, textlines: List[Quadrilateral], src_w: int, src_h: int, dst_w: int, dst_h: int) -> List[Quadrilateral]:
        """Scale quadrilateral coordinates from source resolution to destination resolution."""
        if not textlines:
            return textlines
        
        scale_x = dst_w / src_w
        scale_y = dst_h / src_h
        scaled_textlines = []
        
        for tl in textlines:
            scaled_pts = np.round(tl.pts * np.array([scale_x, scale_y])).astype(np.int32)
            scaled_tl = Quadrilateral(scaled_pts, tl.text, tl.prob)
            scaled_textlines.append(scaled_tl)
            
        return scaled_textlines