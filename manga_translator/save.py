import os
from PIL import Image
from abc import abstractmethod

from .utils import Context


class FormatNotSupportedException(Exception):
    def __init__(self, fmt: str):
        super().__init__(f'Format {fmt} is not supported.')

OUTPUT_FORMATS = {}
def register_format(format_cls):
    for fmt in format_cls.SUPPORTED_FORMATS:
        if fmt in OUTPUT_FORMATS:
            raise Exception(f'Tried to register multiple ExportFormats for "{fmt}"')
        OUTPUT_FORMATS[fmt] = format_cls()
    return format_cls

class ExportFormat():
    SUPPORTED_FORMATS = []

    # Subclasses will be auto registered
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        register_format(cls)

    def save(self, result: Image.Image, dest: str, ctx: Context):
        self._save(result, dest, ctx)

    @abstractmethod
    def _save(self, result: Image.Image, dest: str, ctx: Context):
        pass

def save_result(result: Image.Image, dest: str, ctx: Context):
    _, ext = os.path.splitext(dest)
    ext = ext[1:]
    if ext not in OUTPUT_FORMATS:
        raise FormatNotSupportedException(ext)

    format_handler: ExportFormat = OUTPUT_FORMATS[ext]
    format_handler.save(result, dest, ctx)


# -- Format Implementations

class ImageFormat(ExportFormat):
    SUPPORTED_FORMATS = ['png', 'webp']

    def _save(self, result: Image.Image, dest: str, ctx: Context):
        result.save(dest)

class JPGFormat(ExportFormat):
    SUPPORTED_FORMATS = ['jpg']

    def _save(self, result: Image.Image, dest: str, ctx: Context):
        result = result.convert('RGB')
        result.save(dest, quality=ctx.save_quality)

# class KraFormat(ExportFormat):
#     SUPPORTED_FORMATS = ['kra']

#     def _save(self, result: Image.Image, dest: str, ctx: Context):
#         ...

# class SvgFormat(TranslationExportFormat):
#     SUPPORTED_FORMATS = ['svg']

