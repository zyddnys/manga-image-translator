import os
from PIL import Image
from abc import abstractmethod

from .utils import Context


class FormatNotSupportedException(Exception):
    def __init__(self, fmt: str):
        super().__init__(f'Format {fmt} is not supported.')

# Auto-register subclasses
OUTPUT_FORMATS = {}
def register_format(format_obj):
    for fmt in format_obj.SUPPORTED_FORMATS:
        OUTPUT_FORMATS[fmt] = format_obj
    return format_obj

class SubClassRegister(type):
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        register_format(cls)


class TranslationExportFormat(SubClassRegister):
    SUPPORTED_FORMATS = []

    def save(self, result: Image.Image, dest: str, ctx: Context):
        self._save(result, dest, ctx)

    @abstractmethod
    def _save(self, result: Image.Image, dest: str, ctx: Context):
        pass

def save_result(result: Image.Image, dest: str, ctx: Context):
    _, ext = os.path.splitext(dest)
    ext = ext[1:]
    # result.save(dest)
    if ext not in OUTPUT_FORMATS:
        raise FormatNotSupportedException(ext)
    format_handler = OUTPUT_FORMATS[ext]
    format_handler.save(result, dest, ctx)


# -- Format Implementations

class ImageFormat(TranslationExportFormat):
    SUPPORTED_FORMATS = ['png', 'jpg', 'webp']

    def _save(self, result: Image.Image, dest: str, ctx: Context):
        ctx.result.save(dest)

# class KraFormat(TranslationExportFormat):
#     SUPPORTED_FORMATS = ['kra']

#     def _save(self, result: Image.Image, dest: str, ctx: Context):
#         ...

# class SvgFormat(TranslationExportFormat):
#     SUPPORTED_FORMATS = ['svg']

