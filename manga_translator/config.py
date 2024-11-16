from enum import Enum, IntEnum

from pydantic import BaseModel
from typing import Optional

class Renderer(IntEnum):
    default = 0
    manga2Eng = 1

class Alignment(IntEnum):
    auto = 0
    left = 1
    center = 2
    right = 3

class Direction(IntEnum):
    auto = 0
    h = 1
    v = 2

class InpaintPrecision(IntEnum):
    fp32 = 0
    fp16 = 1
    bf16 = 2

class RenderConfig(BaseModel):
    renderer: Renderer = Renderer.default
    """Render english text translated from manga with some additional typesetting. Ignores some other argument options"""
    alignment: Alignment = Alignment.auto
    """Align rendered text"""
    disable_font_border: bool = False
    """Disable font border"""
    font_size_offset: int = 0
    """Offset font size by a given amount, positive number increase font size and vice versa"""
    font_size_minimum: int = -1
    """Minimum output font size. Default is image_sides_sum/200"""
    direction: Direction = Direction.auto
    """Force text to be rendered horizontally/vertically/none"""
    uppercase: bool = False
    """Change text to uppercase"""
    lowercase: bool = False
    """Change text to lowercase"""
    gimp_font: str = 'Sans-serif'
    """Font family to use for gimp rendering."""
    no_hyphenation: bool = False
    """If renderer should be splitting up words using a hyphen character (-)"""
    font_color: Optional[str] = None
    """Overwrite the text fg/bg color detected by the OCR model. Use hex string without the "#" such as FFFFFF for a white foreground or FFFFFF:000000 to also have a black background around the text."""
    line_spacing: Optional[float] = None
    """Line spacing is font_size * this value. Default is 0.01 for horizontal text and 0.2 for vertical."""
    font_size: Optional[int] = None
    """Use fixed font size for rendering"""


class UpscaleConfig(BaseModel):
    upscaler: str = 'esrgan' #todo: validate UPSCALERS #todo: convert to enum
    """Upscaler to use. --upscale-ratio has to be set for it to take effect"""
    revert_upscaling: bool = False
    """Downscales the previously upscaled image after translation back to original size (Use with --upscale-ratio)."""
    upscale_ratio: Optional[float] = None
    """Image upscale ratio applied before detection. Can improve text detection."""

class TranslatorConfig(BaseModel):
    translator: str = "google" #todo: validate TRANSLATORS todo: convert to enum
    """Language translator to use"""
    target_lang: str = 'ENG' #todo: validate VALID_LANGUAGES #todo: convert to enum
    """Destination language"""
    no_text_lang_skip: bool = False
    """Dont skip text that is seemingly already in the target language."""
    skip_lang: Optional[str] = None
    """Skip translation if source image is one of the provide languages, use comma to separate multiple languages. Example: JPN,ENG"""
    gpt_config: Optional[str] = None  # todo: no more path
    """Path to GPT config file, more info in README"""
    translator_chain: Optional[str] = None  # todo: add parser translator_chain #todo: merge into one
    """Output of one translator goes in another. Example: --translator-chain "google:JPN;sugoi:ENG"."""
    selective_translation: Optional[str] = None  # todo: add parser translator_chain #todo: merge into one
    """Select a translator based on detected language in image. Note the first translation service acts as default if the language isn\'t defined. Example: --translator-chain "google:JPN;sugoi:ENG".'"""

class DetectorConfig(BaseModel):
    """"""
    detector: str = 'default' #todo: validate DETECTORS #todo: convert to enum
    """"Text detector used for creating a text mask from an image, DO NOT use craft for manga, it\'s not designed for it"""
    detection_size: int = 1536
    """Size of image used for detection"""
    text_threshold: float = 0.5
    """Threshold for text detection"""
    det_rotate: bool = False
    """Rotate the image for detection. Might improve detection."""
    det_auto_rotate: bool = False
    """Rotate the image for detection to prefer vertical textlines. Might improve detection."""
    det_invert: bool = False
    """Invert the image colors for detection. Might improve detection."""
    det_gamma_correct: bool = False
    """Applies gamma correction for detection. Might improve detection."""
    ignore_bubble: int = 0
    """The threshold for ignoring text in non bubble areas, with valid values ranging from 1 to 50, does not ignore others. Recommendation 5 to 10. If it is too low, normal bubble areas may be ignored, and if it is too large, non bubble areas may be considered normal bubbles"""

class InpainterConfig(BaseModel):
    inpainter: str = 'lama_large' #todo: validate INPAINTERS  #todo: convert to enum
    """Inpainting model to use"""
    inpainting_size: int = 2048
    """Size of image used for inpainting (too large will result in OOM)"""
    inpainting_precision: InpaintPrecision = InpaintPrecision.fp32
    """Inpainting precision for lama, use bf16 while you can."""


class ColorizerConfig(BaseModel):
    colorization_size: int = 576
    """Size of image used for colorization. Set to -1 to use full image size"""
    denoise_sigma: int = 30
    """Used by colorizer and affects color strength, range from 0 to 255 (default 30). -1 turns it off."""
    colorizer: Optional[str] = None  # todo: validate COLORIZERS  #todo: convert to enum
    """Colorization model to use."""



class OcrConfig(BaseModel):
    use_mocr_merge: bool = False
    """Use bbox merge when Manga OCR inference."""
    ocr: str = '48px' #todo: validate OCRS #todo: convert to enum
    """Optical character recognition (OCR) model to use"""
    min_text_length: int = 0
    """Minimum text length of a text region"""

class Config(BaseModel):
    # unclear
    pre_dict: Optional[str] = None
    post_dict: Optional[str] = None

    # json
    filter_text: Optional[str] = None
    """Filter regions by their text with a regex. Example usage: '.*badtext.*'"""
    render: RenderConfig
    """render configs"""
    upscale: UpscaleConfig
    """upscaler configs"""
    translator: TranslatorConfig
    """tanslator configs"""
    detector: DetectorConfig
    """detector configs"""
    colorizer: ColorizerConfig
    """colorizer configs"""
    inpainter: InpainterConfig
    """inpainter configs"""
    ocr: OcrConfig
    """Ocr configs"""
    # ?
    unclip_ratio: float = 2.3
    """How much to extend text skeleton to form bounding box"""
    kernel_size: int = 3
    """Set the convolution kernel size of the text erasure area to completely clean up text residues"""
    mask_dilation_offset: int = 0
    """By how much to extend the text mask to remove left-over text pixels of the original image."""
    box_threshold: float = 0.7
    """Threshold for bbox generation"""
