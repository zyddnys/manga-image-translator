import argparse
import re
from enum import Enum

from typing import Optional

from omegaconf import OmegaConf
from pydantic import BaseModel


# TODO: Refactor
class TranslatorChain:
    def __init__(self, string: str):
        """
        Parses string in form 'trans1:lang1;trans2:lang2' into chains,
        which will be executed one after another when passed to the dispatch function.
        """
        from manga_translator.translators import TRANSLATORS, VALID_LANGUAGES
        if not string:
            raise Exception('Invalid translator chain')
        self.chain = []
        self.target_lang = None
        for g in string.split(';'):
            trans, lang = g.split(':')
            translator = Translator[trans]
            if translator not in TRANSLATORS:
                raise ValueError(f'Invalid choice: %s (choose from %s)' % (trans, ', '.join(map(repr, TRANSLATORS))))
            if lang not in VALID_LANGUAGES:
                raise ValueError(f'Invalid choice: %s (choose from %s)' % (lang, ', '.join(map(repr, VALID_LANGUAGES))))
            self.chain.append((translator, lang))
        self.translators, self.langs = list(zip(*self.chain))

    def has_offline(self) -> bool:
        """
        Returns True if the chain contains offline translators.
        """
        from manga_translator.translators import OFFLINE_TRANSLATORS
        return any(translator in OFFLINE_TRANSLATORS for translator in self.translators)

    def __eq__(self, __o: object) -> bool:
        if type(__o) is str:
            return __o == self.translators[0]
        return super.__eq__(self, __o)


def translator_chain(string):
    try:
        return TranslatorChain(string)
    except ValueError as e:
        raise argparse.ArgumentTypeError(e)
    except Exception:
        raise argparse.ArgumentTypeError(f'Invalid translator_chain value: "{string}". Example usage: --translator "google:sugoi" -l "JPN:ENG"')


def hex2rgb(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

class Renderer(str, Enum):
    default = "default"
    manga2Eng = "manga2eng"
    none = "none"

class Alignment(str, Enum):
    auto = "auto"
    left = "left"
    center = "center"
    right = "right"

class Direction(str, Enum):
    auto = "auto"
    h = "horizontal"
    v = "vertical"

class InpaintPrecision(str, Enum):
    fp32 = "fp32"
    fp16 = "fp16"
    bf16 = "bf16"

    def __str__(self):
        return self.name

class Detector(str, Enum):
    default = "default"
    dbconvnext = "dbconvnext"
    ctd = "ctd"
    craft = "craft"
    paddle = "paddle"
    none = "none"

class Inpainter(str, Enum):
    default = "default"
    lama_large = "lama_large"
    lama_mpe = "lama_mpe"
    sd = "sd"
    none = "none"
    original = "original"

class Colorizer(str, Enum):
    none = "none"
    mc2 = "mc2"

class Ocr(str, Enum):
    ocr32px = "32px"
    ocr48px = "48px"
    ocr48px_ctc = "48px_ctc"
    mocr = "mocr"

class Translator(str, Enum):
    youdao = "youdao"
    baidu = "baidu"
    deepl = "deepl"
    papago = "papago"
    caiyun = "caiyun"
    chatgpt = "chatgpt"
    none = "none"
    original = "original"
    sakura = "sakura"
    deepseek = "deepseek"
    groq = "groq"
    gemini = "gemini"
    custom_openai = "custom_openai"
    offline = "offline"
    nllb = "nllb"
    nllb_big = "nllb_big"
    sugoi = "sugoi"
    jparacrawl = "jparacrawl"
    jparacrawl_big = "jparacrawl_big"
    m2m100 = "m2m100"
    m2m100_big = "m2m100_big"
    mbart50 = "mbart50"
    qwen2 = "qwen2"
    qwen2_big = "qwen2_big"

    def __str__(self):
        return self.name

    # Map 'openai' and any translator starting with 'gpt'* to 'chatgpt'
    @classmethod
    def _missing_(cls, value):
        if value.startswith('gpt') or value == 'openai':
            return cls.chatgpt
        raise ValueError(f"{value} is not a valid {cls.__name__}")


class Upscaler(str, Enum):
    waifu2x = "waifu2x"
    esrgan = "esrgan"
    upscler4xultrasharp = "4xultrasharp"

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
    line_spacing: Optional[int] = None
    """Line spacing is font_size * this value. Default is 0.01 for horizontal text and 0.2 for vertical."""
    font_size: Optional[int] = None
    """Use fixed font size for rendering"""
    _font_color_fg = None
    _font_color_bg = None
    @property
    def font_color_fg(self):
        if self.font_color and not self._font_color_fg:
            colors = self.font_color.split(':')
            try:
                self._font_color_fg = hex2rgb(colors[0])
                self._font_color_bg = hex2rgb(colors[1]) if len(colors) > 1 else None
            except:
                raise Exception(
                    f'Invalid --font-color value: {self.font_color}. Use a hex value such as FF0000')
        return self._font_color_fg

    @property
    def font_color_bg(self):
        if self.font_color and not self._font_color_bg:
            colors = self.font_color.split(':')
            try:
                self._font_color_fg = hex2rgb(colors[0])
                self._font_color_bg = hex2rgb(colors[1]) if len(colors) > 1 else None
            except:
                raise Exception(
                    f'Invalid --font-color value: {self.font_color}. Use a hex value such as FF0000')
        return self._font_color_bg

class UpscaleConfig(BaseModel):
    upscaler: Upscaler = Upscaler.esrgan
    """Upscaler to use. --upscale-ratio has to be set for it to take effect"""
    revert_upscaling: bool = False
    """Downscales the previously upscaled image after translation back to original size (Use with --upscale-ratio)."""
    upscale_ratio: Optional[int] = None
    """Image upscale ratio applied before detection. Can improve text detection."""

class TranslatorConfig(BaseModel):
    translator: Translator = Translator.sugoi
    """Language translator to use"""
    target_lang: str = 'ENG' #todo: validate VALID_LANGUAGES #todo: convert to enum
    """Destination language"""
    no_text_lang_skip: bool = False
    """Dont skip text that is seemingly already in the target language."""
    skip_lang: Optional[str] = None
    """Skip translation if source image is one of the provide languages, use comma to separate multiple languages. Example: JPN,ENG"""
    gpt_config: Optional[str] = None  # todo: no more path
    """Path to GPT config file, more info in README"""
    translator_chain: Optional[str] = None
    """Output of one translator goes in another. Example: --translator-chain "google:JPN;sugoi:ENG"."""
    selective_translation: Optional[str] = None
    """Select a translator based on detected language in image. Note the first translation service acts as default if the language isn\'t defined. Example: --translator-chain "google:JPN;sugoi:ENG".'"""
    _translator_gen = None
    _gpt_config = None

    @property
    def translator_gen(self):
        if self._translator_gen is None:
            if self.selective_translation is not None:
                #todo: refactor TranslatorChain
                trans =  translator_chain(self.selective_translation)
                trans.target_lang = self.target_lang
                self._translator_gen = trans
            elif self.translator_chain is not None:
                trans = translator_chain(self.translator_chain)
                trans.target_lang = trans.langs[0]
                self._translator_gen = trans
            else:
                self._translator_gen = TranslatorChain(f'{str(self.translator)}:{self.target_lang}')
        return self._translator_gen

    @property
    def chatgpt_config(self):
        if self.gpt_config is not None and self._gpt_config is None:
            #todo: load from already loaded file
            self._gpt_config = OmegaConf.load(self.gpt_config)
        return self._gpt_config


class DetectorConfig(BaseModel):
    """"""
    detector: Detector =Detector.default
    """"Text detector used for creating a text mask from an image, DO NOT use craft for manga, it\'s not designed for it"""
    detection_size: int = 2048
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
    box_threshold: float = 0.7
    """Threshold for bbox generation"""
    unclip_ratio: float = 2.3
    """How much to extend text skeleton to form bounding box"""

class InpainterConfig(BaseModel):
    inpainter: Inpainter = Inpainter.lama_large
    """Inpainting model to use"""
    inpainting_size: int = 2048
    """Size of image used for inpainting (too large will result in OOM)"""
    inpainting_precision: InpaintPrecision = InpaintPrecision.bf16
    """Inpainting precision for lama, use bf16 while you can."""

class ColorizerConfig(BaseModel):
    colorization_size: int = 576
    """Size of image used for colorization. Set to -1 to use full image size"""
    denoise_sigma: int = 30
    """Used by colorizer and affects color strength, range from 0 to 255 (default 30). -1 turns it off."""
    colorizer: Colorizer = Colorizer.none
    """Colorization model to use."""

class OcrConfig(BaseModel):
    use_mocr_merge: bool = False
    """Use bbox merge when Manga OCR inference."""
    ocr: Ocr = Ocr.ocr48px
    """Optical character recognition (OCR) model to use"""
    min_text_length: int = 0
    """Minimum text length of a text region"""
    ignore_bubble: int = 0
    """The threshold for ignoring text in non bubble areas, with valid values ranging from 1 to 50, does not ignore others. Recommendation 5 to 10. If it is too low, normal bubble areas may be ignored, and if it is too large, non bubble areas may be considered normal bubbles"""

class Config(BaseModel):
    filter_text: Optional[str] = None
    """Filter regions by their text with a regex. Example usage: '.*badtext.*'"""
    render: RenderConfig = RenderConfig()
    """render configs"""
    upscale: UpscaleConfig = UpscaleConfig()
    """upscaler configs"""
    translator: TranslatorConfig = TranslatorConfig()
    """tanslator configs"""
    detector: DetectorConfig = DetectorConfig()
    """detector configs"""
    colorizer: ColorizerConfig = ColorizerConfig()
    """colorizer configs"""
    inpainter: InpainterConfig = InpainterConfig()
    """inpainter configs"""
    ocr: OcrConfig = OcrConfig()
    """Ocr configs"""
    # ?
    kernel_size: int = 3
    """Set the convolution kernel size of the text erasure area to completely clean up text residues"""
    mask_dilation_offset: int = 0
    """By how much to extend the text mask to remove left-over text pixels of the original image."""
    _filter_text = None

    @property
    def re_filter_text(self):
        if self._filter_text is None:
            self._filter_text = re.compile(self.filter_text)
        return self._filter_text
