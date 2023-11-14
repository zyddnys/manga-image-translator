import argparse
import os
from urllib.parse import unquote

from .detection import DETECTORS
from .ocr import OCRS
from .inpainting import INPAINTERS
from .translators import VALID_LANGUAGES, TRANSLATORS, TranslatorChain
from .upscaling import UPSCALERS
from .colorization import COLORIZERS
from .save import OUTPUT_FORMATS

def url_decode(s):
    s = unquote(s)
    if s.startswith('file:///'):
        s = s[len('file://'):]
    return s

# Additional argparse types
def path(string):
    if not string:
        return ''
    s = url_decode(os.path.expanduser(string))
    if not os.path.exists(s):
        raise argparse.ArgumentTypeError(f'No such file or directory: "{string}"')
    return s

def file_path(string):
    if not string:
        return ''
    s = url_decode(os.path.expanduser(string))
    if not os.path.exists(s):
        raise argparse.ArgumentTypeError(f'No such file: "{string}"')
    return s

def dir_path(string):
    if not string:
        return ''
    s = url_decode(os.path.expanduser(string))
    if not os.path.exists(s):
        raise argparse.ArgumentTypeError(f'No such directory: "{string}"')
    return s

# def choice_chain(choices):
#     """Argument type for string chains from choices seperated by ':'. Example: 'choice1:choice2:choice3'"""
#     def _func(string):
#         if choices is not None:
#             for s in string.split(':') or ['']:
#                 if s not in choices:
#                     raise argparse.ArgumentTypeError(f'Invalid choice: %s (choose from %s)' % (s, ', '.join(map(repr, choices))))
#         return string
#     return _func

def translator_chain(string):
    try:
        return TranslatorChain(string)
    except ValueError as e:
        raise argparse.ArgumentTypeError(e)
    except Exception:
        raise argparse.ArgumentTypeError(f'Invalid translator_chain value: "{string}". Example usage: --translator "google:sugoi" -l "JPN:ENG"')


class HelpFormatter(argparse.HelpFormatter):
    INDENT_INCREMENT = 2
    MAX_HELP_POSITION = 24
    WIDTH = None

    def __init__(self, prog: str, indent_increment: int = 2, max_help_position: int = 24, width: int = None):
        super().__init__(prog, self.INDENT_INCREMENT, self.MAX_HELP_POSITION, self.WIDTH)

    def _format_action_invocation(self, action: argparse.Action) -> str:
        if action.option_strings:

            # if the Optional doesn't take a value, format is:
            #    -s, --long
            if action.nargs == 0:
                return ', '.join(action.option_strings)

            # if the Optional takes a value, format is:
            #    -s, --long ARGS
            else:
                default = self._get_default_metavar_for_optional(action)
                args_string = self._format_args(action, default)
                return ', '.join(action.option_strings) + ' ' + args_string
        else:
            return super()._format_action_invocation(action)


parser = argparse.ArgumentParser(prog='manga_translator', description='Seamlessly translate mangas into a chosen language', formatter_class=HelpFormatter)
parser.add_argument('-m', '--mode', default='batch', type=str, choices=['demo', 'batch', 'web', 'web_client', 'ws', 'api'], help='Run demo in single image demo mode (demo), batch translation mode (batch), web service mode (web)')
parser.add_argument('-i', '--input', default=None, type=path, nargs='+', help='Path to an image file if using demo mode, or path to an image folder if using batch mode')
parser.add_argument('-o', '--dest', default='', type=str, help='Path to the destination folder for translated images in batch mode')
parser.add_argument('-l', '--target-lang', default='CHS', type=str, choices=VALID_LANGUAGES, help='Destination language')
parser.add_argument('-v', '--verbose', action='store_true', help='Print debug info and save intermediate images in result folder')
parser.add_argument('-f', '--format', default=None, choices=OUTPUT_FORMATS, help='Output format of the translation.')
parser.add_argument('--attempts', default=0, type=int, help='Retry attempts on encountered error. -1 means infinite times.')
parser.add_argument('--ignore-errors', action='store_true', help='Skip image on encountered error.')
parser.add_argument('--overwrite', action='store_true', help='Overwrite already translated images in batch mode.')
parser.add_argument('--skip-no-text', action='store_true', help='Skip image without text (Will not be saved).')
parser.add_argument('--model-dir', default=None, type=dir_path, help='Model directory (by default ./models in project root)')

g = parser.add_mutually_exclusive_group()
g.add_argument('--use-cuda', action='store_true', help='Turn on/off cuda')
g.add_argument('--use-cuda-limited', action='store_true', help='Turn on/off cuda (excluding offline translator)')

parser.add_argument('--detector', default='default', type=str, choices=DETECTORS, help='Text detector used for creating a text mask from an image, DO NOT use craft for manga, it\'s not designed for it')
parser.add_argument('--ocr', default='48px', type=str, choices=OCRS, help='Optical character recognition (OCR) model to use')
parser.add_argument('--inpainter', default='lama_large', type=str, choices=INPAINTERS, help='Inpainting model to use')
parser.add_argument('--upscaler', default='esrgan', type=str, choices=UPSCALERS, help='Upscaler to use. --upscale-ratio has to be set for it to take effect')
parser.add_argument('--upscale-ratio', default=None, type=float, help='Image upscale ratio applied before detection. Can improve text detection.')
parser.add_argument('--colorizer', default=None, type=str, choices=COLORIZERS, help='Colorization model to use.')

g = parser.add_mutually_exclusive_group()
g.add_argument('--translator', default='google', type=str, choices=TRANSLATORS, help='Language translator to use')
g.add_argument('--translator-chain', default=None, type=translator_chain, help='Output of one translator goes in another. Example: --translator-chain "google:JPN;sugoi:ENG".')
g.add_argument('--selective-translation', default=None, type=translator_chain, help='Select a translator based on detected language in image. Note the first translation service acts as default if the language isnt defined. Example: --translator-chain "google:JPN;sugoi:ENG".')

parser.add_argument('--revert-upscaling', action='store_true', help='Downscales the previously upscaled image after translation back to original size (Use with --upscale-ratio).')
parser.add_argument('--detection-size', default=1536, type=int, help='Size of image used for detection')
parser.add_argument('--det-rotate', action='store_true', help='Rotate the image for detection. Might improve detection.')
parser.add_argument('--det-auto-rotate', action='store_true', help='Rotate the image for detection to prefer vertical textlines. Might improve detection.')
parser.add_argument('--det-invert', action='store_true', help='Invert the image colors for detection. Might improve detection.')
parser.add_argument('--det-gamma-correct', action='store_true', help='Applies gamma correction for detection. Might improve detection.')
parser.add_argument('--unclip-ratio', default=2.3, type=float, help='How much to extend text skeleton to form bounding box')
parser.add_argument('--box-threshold', default=0.7, type=float, help='Threshold for bbox generation')
parser.add_argument('--text-threshold', default=0.5, type=float, help='Threshold for text detection')
parser.add_argument('--min-text-length', default=0, type=int, help='Minimum text length of a text region')
parser.add_argument('--inpainting-size', default=2048, type=int, help='Size of image used for inpainting (too large will result in OOM)')
parser.add_argument('--inpainting-precision', default='fp32', type=str, help='Inpainting precision for lama, use bf16 while you can.', choices=['fp32', 'fp16', 'bf16'])
parser.add_argument('--colorization-size', default=576, type=int, help='Size of image used for colorization. Set to -1 to use full image size')
parser.add_argument('--denoise-sigma', default=30, type=int, help='Used by colorizer and affects color strength, range from 0 to 255 (default 30). -1 turns it off.')
parser.add_argument('--mask-dilation-offset', default=0, type=int, help='By how much to extend the text mask to remove left-over text pixels of the original image.')

parser.add_argument('--font-size', default=None, type=int, help='Use fixed font size for rendering')
parser.add_argument('--font-size-offset', default=0, type=int, help='Offset font size by a given amount, positive number increase font size and vice versa')
parser.add_argument('--font-size-minimum', default=-1, type=int, help='Minimum output font size. Default is image_sides_sum/200')
parser.add_argument('--font-color', default=None, type=str, help='Overwrite the text fg/bg color detected by the OCR model. Use hex string without the "#" such as FFFFFF for a white foreground or FFFFFF:000000 to also have a black background around the text.')
parser.add_argument('--line-spacing', default=None, type=float, help='Line spacing is font_size * this value. Default is 0.01 for horizontal text and 0.2 for vertical.')

g = parser.add_mutually_exclusive_group()
g.add_argument('--force-horizontal', action='store_true', help='Force text to be rendered horizontally')
g.add_argument('--force-vertical', action='store_true', help='Force text to be rendered vertically')

g = parser.add_mutually_exclusive_group()
g.add_argument('--align-left', action='store_true', help='Align rendered text left')
g.add_argument('--align-center', action='store_true', help='Align rendered text centered')
g.add_argument('--align-right', action='store_true', help='Align rendered text right')

g = parser.add_mutually_exclusive_group()
g.add_argument('--uppercase', action='store_true', help='Change text to uppercase')
g.add_argument('--lowercase', action='store_true', help='Change text to lowercase')

parser.add_argument('--no-hyphenation', action='store_true', help='If renderer should be splitting up words using a hyphen character (-)')
parser.add_argument('--manga2eng', action='store_true', help='Render english text translated from manga with some additional typesetting. Ignores some other argument options')
parser.add_argument('--gpt-config', type=file_path, help='Path to GPT config file, more info in README')
parser.add_argument('--use-mtpe', action='store_true', help='Turn on/off machine translation post editing (MTPE) on the command line (works only on linux right now)')

g = parser.add_mutually_exclusive_group()
g.add_argument('--save-text', action='store_true', help='Save extracted text and translations into a text file.')
g.add_argument('--save-text-file', default='', type=str, help='Like --save-text but with a specified file path.')

parser.add_argument('--filter-text', default=None, type=str, help='Filter regions by their text with a regex. Example usage: --text-filter ".*badtext.*"')
parser.add_argument('--prep-manual', action='store_true', help='Prepare for manual typesetting by outputting blank, inpainted images, plus copies of the original for reference')
parser.add_argument('--font-path', default='', type=file_path, help='Path to font file')
parser.add_argument('--gimp-font', default='Sans-serif', type=str, help='Font family to use for gimp rendering.')
parser.add_argument('--host', default='127.0.0.1', type=str, help='Used by web module to decide which host to attach to')
parser.add_argument('--port', default=5003, type=int, help='Used by web module to decide which port to attach to')
parser.add_argument('--nonce', default=os.getenv('MT_WEB_NONCE', ''), type=str, help='Used by web module as secret for securing internal web server communication')
# parser.add_argument('--log-web', action='store_true', help='Used by web module to decide if web logs should be surfaced')
parser.add_argument('--ws-url', default='ws://localhost:5000', type=str, help='Server URL for WebSocket mode')
parser.add_argument('--save-quality', default=100, type=int, help='Quality of saved JPEG image, range from 0 to 100 with 100 being best')
parser.add_argument('--ignore-bubble', default=0, type=int, help='The threshold for ignoring text in non bubble areas, with valid values ranging from 1 to 50, does not ignore others. Recommendation 5 to 10. If it is too low, normal bubble areas may be ignored, and if it is too large, non bubble areas may be considered normal bubbles')

# Generares dict with a default value for each argument
DEFAULT_ARGS = vars(parser.parse_args([]))
