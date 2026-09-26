"""
SD1.5 + ControlNet Inpainter for manga-image-translator.

Models used:
  Base checkpoint : Meina/MeinaMix_V11  (SD1.5 manga, HuggingFace)
  ControlNet      : lllyasviel/control_v11p_sd15_inpaint   (inpaint guidance)
  ControlNet      : lllyasviel/control_v11p_sd15_lineart   (lineart preservation)
  VAE             : stabilityai/sd-vae-ft-mse (sharpens output)

16GB VRAM optimisations:
  - xformers attention  (--xformers)
  - torch.float16 weights  (--half / fp16)
  - enable_vae_slicing + enable_model_cpu_offload as fallback
  - Sequential CPU offload NOT used by default (too slow for batch)

Plugs into MIT's OfflineInpainter so download / load / unload are handled
by the existing pipeline without any changes to manga_translator.py.
"""

from __future__ import annotations

import asyncio
import os
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image

from .common import OfflineInpainter
from ..config import InpainterConfig
from ..utils import resize_keep_aspect


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _np_to_pil(arr: np.ndarray) -> Image.Image:
    """RGB uint8 numpy → PIL."""
    return Image.fromarray(arr.astype(np.uint8))


def _pil_to_np(img: Image.Image) -> np.ndarray:
    return np.array(img)


def _make_inpaint_condition(image: Image.Image, mask: Image.Image) -> Image.Image:
    """
    Build the inpaint ControlNet conditioning image:
    masked-out pixels → 0.5 grey  (as per lllyasviel spec).
    """
    image_np = np.array(image).astype(np.float32) / 255.0
    mask_np = np.array(mask.convert('L')).astype(np.float32) / 255.0
    mask_np = (mask_np > 0.5).astype(np.float32)
    image_np[mask_np > 0.5] = -1.0          # -1 = masked region
    image_np = np.clip(image_np, -1, 1)
    image_np = (image_np + 1.0) / 2.0       # back to [0,1]
    return Image.fromarray((image_np * 255).astype(np.uint8))


def _make_lineart_condition(image: Image.Image) -> Image.Image:
    """
    Simple Canny-based lineart map for ControlNet-lineart conditioning.
    For production quality use ControlNet's own lineart detector;
    Canny is used here to avoid the extra model download dependency.
    """
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    # Invert: white lines on black bg  →  ControlNet lineart expects this
    edges = cv2.bitwise_not(edges)
    return Image.fromarray(cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB))


def _pad_to_multiple(img: np.ndarray, multiple: int = 8) -> tuple[np.ndarray, int, int]:
    h, w = img.shape[:2]
    new_h = ((h + multiple - 1) // multiple) * multiple
    new_w = ((w + multiple - 1) // multiple) * multiple
    padded = cv2.copyMakeBorder(
        img, 0, new_h - h, 0, new_w - w, cv2.BORDER_REFLECT_101
    )
    return padded, h, w


# ---------------------------------------------------------------------------
# Inpainter
# ---------------------------------------------------------------------------

class MangaSDControlNetInpainter(OfflineInpainter):
    """
    Stable Diffusion 1.5 + dual ControlNet inpainter.
    Registered as Inpainter.manga_sd_cn.
    """

    # No binary weight download needed — diffusers pulls from HuggingFace hub.
    # We still need _MODEL_MAPPING to satisfy ModelWrapper.__init__ check.
    _MODEL_MAPPING = {}

    # HuggingFace model IDs  (change to local paths if offline)
    BASE_MODEL_ID   = 'Meina/MeinaMix_V11'
    CN_INPAINT_ID   = 'lllyasviel/control_v11p_sd15_inpaint'
    CN_LINEART_ID   = 'lllyasviel/control_v11p_sd15_lineart'
    VAE_ID          = 'stabilityai/sd-vae-ft-mse'

    # Generation params
    NUM_INFERENCE_STEPS = 30
    GUIDANCE_SCALE      = 7.5
    CN_INPAINT_SCALE    = 1.0
    CN_LINEART_SCALE    = 0.65   # softer — preserves lineart without hard edges
    MAX_INPAINT_SIZE    = 768    # px long edge; fits 16GB VRAM in fp16

    POSITIVE_PROMPT = (
        'masterpiece, best quality, manga, monochrome, lineart, '
        'clean screentones, no text, no speech bubbles'
    )
    NEGATIVE_PROMPT = (
        'text, speech bubble, word, letter, watermark, lowres, '
        'bad anatomy, worst quality, blurry, jpeg artifacts, '
        'extra lines, overexposed'
    )

    def __init__(self, *args, **kwargs):
        os.makedirs(self.model_dir, exist_ok=True)
        super().__init__(*args, **kwargs)

    # ------------------------------------------------------------------
    # Load / Unload
    # ------------------------------------------------------------------

    async def _load(self, device: str):
        from diffusers import (
            ControlNetModel,
            StableDiffusionControlNetInpaintPipeline,
            AutoencoderKL,
        )

        self.device = device
        dtype = torch.float16 if device.startswith('cuda') else torch.float32

        self.logger.info('Loading ControlNet models …')
        cn_inpaint = ControlNetModel.from_pretrained(
            self.CN_INPAINT_ID, torch_dtype=dtype
        )
        cn_lineart = ControlNetModel.from_pretrained(
            self.CN_LINEART_ID, torch_dtype=dtype
        )

        self.logger.info(f'Loading base checkpoint: {self.BASE_MODEL_ID} …')
        vae = AutoencoderKL.from_pretrained(self.VAE_ID, torch_dtype=dtype)

        self.pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            self.BASE_MODEL_ID,
            controlnet=[cn_inpaint, cn_lineart],
            vae=vae,
            torch_dtype=dtype,
            safety_checker=None,
            requires_safety_checker=False,
        )

        # xformers (import guard — fails gracefully if not installed)
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
            self.logger.info('xformers enabled')
        except Exception:
            self.logger.warning('xformers not available — falling back to default attn')

        self.pipe.enable_vae_slicing()

        # Try to move fully to GPU; fall back to CPU offload if OOM
        try:
            self.pipe.to(device)
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                self.logger.warning('VRAM full — enabling model CPU offload')
                self.pipe.enable_model_cpu_offload()
            else:
                raise

        self.pipe.set_progress_bar_config(disable=True)
        self.logger.info('MangaSDControlNetInpainter ready')

    async def _unload(self):
        del self.pipe

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    async def _infer(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        config: InpainterConfig,
        inpainting_size: int = 1024,
        verbose: bool = False,
    ) -> np.ndarray:

        img_original = image.copy()
        h_orig, w_orig = image.shape[:2]

        # Limit size for VRAM
        work_size = min(inpainting_size, self.MAX_INPAINT_SIZE)

        # Resize to work_size (keep aspect)
        if max(h_orig, w_orig) > work_size:
            image = resize_keep_aspect(image, work_size)
            mask  = resize_keep_aspect(mask, work_size)

        # Pad to multiple of 8 (VAE requirement)
        image_padded, h_work, w_work = _pad_to_multiple(image, 8)
        mask_padded, _, _            = _pad_to_multiple(mask, 8)

        pil_image = _np_to_pil(image_padded)
        pil_mask  = Image.fromarray(mask_padded).convert('L')

        # Build ControlNet conditioning images
        cn_inpaint_image  = _make_inpaint_condition(pil_image, pil_mask)
        cn_lineart_image  = _make_lineart_condition(pil_image)

        # Run pipeline in thread-executor (keeps asyncio loop responsive)
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self.pipe(
                prompt=self.POSITIVE_PROMPT,
                negative_prompt=self.NEGATIVE_PROMPT,
                image=pil_image,
                mask_image=pil_mask,
                control_image=[cn_inpaint_image, cn_lineart_image],
                controlnet_conditioning_scale=[
                    self.CN_INPAINT_SCALE,
                    self.CN_LINEART_SCALE,
                ],
                num_inference_steps=self.NUM_INFERENCE_STEPS,
                guidance_scale=self.GUIDANCE_SCALE,
                height=image_padded.shape[0],
                width=image_padded.shape[1],
                generator=torch.Generator(device=self.device).manual_seed(42),
            )
        )

        # Crop padding and resize back to original dimensions
        inpainted_np = _pil_to_np(result.images[0])
        inpainted_np = inpainted_np[:h_work, :w_work]  # remove padding
        inpainted_np = cv2.resize(
            inpainted_np, (w_orig, h_orig), interpolation=cv2.INTER_LANCZOS4
        )

        # Composite: only fill masked pixels (preserve unmasked original)
        mask_original = cv2.resize(mask, (w_orig, h_orig),
                                   interpolation=cv2.INTER_NEAREST)
        binary_mask = (mask_original > 127).astype(np.float32)[:, :, None]
        output = (
            img_original * (1.0 - binary_mask) +
            inpainted_np * binary_mask
        ).astype(np.uint8)

        return output
