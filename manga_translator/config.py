from pydantic import BaseModel
from typing import Optional

class RenderConfig(BaseModel):
    """"""
    """Render english text translated from manga with some additional typesetting. Ignores some other argument options"""
    renderer: str = 'default' #todo: validate {"default", "manga2eng"} #todo: convert to enum
    """Align rendered text"""
    alignment: str = 'auto'  # todo: validate {'left','center','right'} #todo: convert to enum
    """Disable font border"""
    disable_font_border: bool = False
    """Offset font size by a given amount, positive number increase font size and vice versa"""
    font_size_offset: int = 0
    """Minimum output font size. Default is image_sides_sum/200"""
    font_size_minimum: int = -1
    """Force text to be rendered horizontally/vertically/none"""
    direction: str = 'auto'  # todo: validate {'auto', 'h', 'v'} #todo: convert to enum
    """Change text to uppercase"""
    uppercase: bool = False
    """Change text to lowercase"""
    lowercase: bool = False
    """Font family to use for gimp rendering."""
    gimp_font: str = 'Sans-serif'
    """If renderer should be splitting up words using a hyphen character (-)"""
    no_hyphenation: bool = False
    """Overwrite the text fg/bg color detected by the OCR model. Use hex string without the "#" such as FFFFFF for a white foreground or FFFFFF:000000 to also have a black background around the text."""
    font_color: Optional[str] = None
    """Line spacing is font_size * this value. Default is 0.01 for horizontal text and 0.2 for vertical."""
    line_spacing: Optional[float] = None
    """Use fixed font size for rendering"""
    font_size: Optional[int] = None

class UpscaleConfig(BaseModel):
    """"""
    """Upscaler to use. --upscale-ratio has to be set for it to take effect"""
    upscaler: str = 'esrgan' #todo: validate UPSCALERS #todo: convert to enum
    """Downscales the previously upscaled image after translation back to original size (Use with --upscale-ratio)."""
    revert_upscaling: bool = False
    """Image upscale ratio applied before detection. Can improve text detection."""
    upscale_ratio: Optional[float] = None


class TranslatorConfig(BaseModel):
    """"""
    """Language translator to use"""
    translator: str = "google" #todo: validate TRANSLATORS todo: convert to enum
    """Destination language"""
    target_lang: str = 'ENG' #todo: validate VALID_LANGUAGES #todo: convert to enum
    """Dont skip text that is seemingly already in the target language."""
    no_text_lang_skip: bool = False
    """Skip translation if source image is one of the provide languages, use comma to separate multiple languages. Example: JPN,ENG"""
    skip_lang: Optional[str] = None
    """Path to GPT config file, more info in README"""
    gpt_config: Optional[str] = None  # todo: no more path
    """Output of one translator goes in another. Example: --translator-chain "google:JPN;sugoi:ENG"."""
    translator_chain: Optional[str] = None  # todo: add parser translator_chain #todo: merge into one
    """Select a translator based on detected language in image. Note the first translation service acts as default if the language isn\'t defined. Example: --translator-chain "google:JPN;sugoi:ENG".'"""
    selective_translation: Optional[str] = None  # todo: add parser translator_chain #todo: merge into one

class DetectorConfig(BaseModel):
    """"""
    """"Text detector used for creating a text mask from an image, DO NOT use craft for manga, it\'s not designed for it"""
    detector: str = 'default' #todo: validate DETECTORS #todo: convert to enum
    """Size of image used for detection"""
    detection_size: int = 1536
    """Threshold for text detection"""
    text_threshold: float = 0.5
    """Rotate the image for detection. Might improve detection."""
    det_rotate: bool = False
    """Rotate the image for detection to prefer vertical textlines. Might improve detection."""
    det_auto_rotate: bool = False
    """Invert the image colors for detection. Might improve detection."""
    det_invert: bool = False
    """Applies gamma correction for detection. Might improve detection."""
    det_gamma_correct: bool = False
    """The threshold for ignoring text in non bubble areas, with valid values ranging from 1 to 50, does not ignore others. Recommendation 5 to 10. If it is too low, normal bubble areas may be ignored, and if it is too large, non bubble areas may be considered normal bubbles"""
    ignore_bubble: int = 0

class InpainterConfig(BaseModel):
    """"""
    """Inpainting model to use"""
    inpainter: str = 'lama_large' #todo: validate INPAINTERS  #todo: convert to enum
    """Size of image used for inpainting (too large will result in OOM)"""
    inpainting_size: int = 2048
    """Inpainting precision for lama, use bf16 while you can."""
    inpainting_precision: str = 'fp32' #todo: validate ['fp32', 'fp16', 'bf16'] #todo: convert to enum

class ColorizerConfig(BaseModel):
    """"""
    """Size of image used for colorization. Set to -1 to use full image size"""
    colorization_size: int = 576
    """Used by colorizer and affects color strength, range from 0 to 255 (default 30). -1 turns it off."""
    denoise_sigma: int = 30
    """Colorization model to use."""
    colorizer: Optional[str] = None  # todo: validate COLORIZERS  #todo: convert to enum


class OcrConfig(BaseModel):
    """"""
    """Use bbox merge when Manga OCR inference."""
    use_mocr_merge: bool = False
    """Optical character recognition (OCR) model to use"""
    ocr: str = '48px' #todo: validate OCRS #todo: convert to enum
    """Minimum text length of a text region"""
    min_text_length: int = 0

class Config(BaseModel):
    # unclear
    pre_dict: Optional[str] = None
    post_dict: Optional[str] = None

    # json
    """Filter regions by their text with a regex. Example usage: '.*badtext.*'"""
    filter_text: Optional[str] = None
    """render configs"""
    render: RenderConfig
    """upscaler configs"""
    upscale: UpscaleConfig
    """tanslator configs"""
    translator: TranslatorConfig
    """detector configs"""
    detector: DetectorConfig
    """colorizer configs"""
    colorizer: ColorizerConfig
    """inpainter configs"""
    inpainter: InpainterConfig
    """Ocr configs"""
    ocr: OcrConfig
    # ?
    """How much to extend text skeleton to form bounding box"""
    unclip_ratio: float = 2.3
    """Set the convolution kernel size of the text erasure area to completely clean up text residues"""
    kernel_size: int = 3
    """By how much to extend the text mask to remove left-over text pixels of the original image."""
    mask_dilation_offset: int = 0
    """Threshold for bbox generation"""
    box_threshold: float = 0.7