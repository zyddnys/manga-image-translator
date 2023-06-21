import os
from PIL import Image
from abc import abstractmethod

from .utils import Context


class FormatNotSupportedException(Exception):
    def __init__(self, fmt: str):
        super().__init__(f'Format {fmt} is not supported.')


class ExportFormat(metaclass=ABCMeta):
    SUPPORTED_FORMATS = []

    # Subclasses will be auto registered
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        for fmt in cls.SUPPORTED_FORMATS:
            if fmt in OUTPUT_FORMATS:
                raise Exception(f'Tried to register multiple ExportFormats for "{fmt}"')
            OUTPUT_FORMATS[fmt] = cls

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

@register_format
class ImageFormat(ExportFormat):
    SUPPORTED_FORMATS = ['png', 'webp']

    def _save(self, result: Image.Image, dest: str, ctx: Context):
        result.save(dest)


@register_format
class JPGFormat(ExportFormat):
    SUPPORTED_FORMATS = ['jpg']

    def _save(self, result: Image.Image, dest: str, ctx: Context):
        result = result.convert('RGB')
        # Certain versions of PIL only support JPEG but not JPG
        result.save(dest, quality=ctx.save_quality, format='JPEG')


# class KraFormat(ExportFormat):
#     SUPPORTED_FORMATS = ['kra']

#     def _save(self, result: Image.Image, dest: str, ctx: Context):
#         ...

# class SvgFormat(TranslationExportFormat):
#     SUPPORTED_FORMATS = ['svg']

