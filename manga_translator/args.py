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
#     """Argument type for string chains from choices separated by ':'. Example: 'choice1:choice2:choice3'"""
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
parser.add_argument('-v', '--verbose', action='store_true', help='Print debug info and save intermediate images in result folder')
parser.add_argument('--attempts', default=0, type=int, help='Retry attempts on encountered error. -1 means infinite times.')
parser.add_argument('--ignore-errors', action='store_true', help='Skip image on encountered error.')
parser.add_argument('--model-dir', default=None, type=dir_path, help='Model directory (by default ./models in project root)')
g = parser.add_mutually_exclusive_group()
g.add_argument('--use-gpu', action='store_true', help='Turn on/off gpu (auto switch between mps and cuda)')
g.add_argument('--use-gpu-limited', action='store_true', help='Turn on/off gpu (excluding offline translator)')
parser.add_argument('--font-path', default='', type=file_path, help='Path to font file')
parser.add_argument('--pre-dict', default=None, type=file_path, help='Path to the pre-translation dictionary file')
parser.add_argument('--post-dict', default=None, type=file_path, help='Path to the post-translation dictionary file')

subparsers = parser.add_subparsers(dest='mode', required=True, help='Mode of operation')

# Batch mode
parser_batch = subparsers.add_parser('batch', help='Run in batch translation mode')
parser_batch.add_argument('-i', '--input', required=True, type=dir_path, help='Path to an image folder')
parser_batch.add_argument('-o', '--dest', default='', type=str, help='Path to the destination folder for translated images')
parser_batch.add_argument('-f', '--format', default=None, choices=OUTPUT_FORMATS, help='Output format of the translation.')
parser_batch.add_argument('--overwrite', action='store_true', help='Overwrite already translated images')
parser_batch.add_argument('--skip-no-text', action='store_true', help='Skip image without text (Will not be saved).')
parser_batch.add_argument('--use-mtpe', action='store_true', help='Turn on/off machine translation post editing (MTPE) on the command line (works only on linux right now)')
g_batch = parser_batch.add_mutually_exclusive_group()
g_batch.add_argument('--save-text', action='store_true', help='Save extracted text and translations into a text file.')
g_batch.add_argument('--save-text-file', default='', type=str, help='Like --save-text but with a specified file path.')
parser_batch.add_argument('--prep-manual', action='store_true', help='Prepare for manual typesetting by outputting blank, inpainted images, plus copies of the original for reference')
parser_batch.add_argument('--save-quality', default=100, type=int, help='Quality of saved JPEG image, range from 0 to 100 with 100 being best')

# Web mode
parser_web = subparsers.add_parser('web', help='Run in web service mode')
parser_web.add_argument('--host', default='127.0.0.1', type=str, help='Host for web service')
parser_web.add_argument('--port', default=5003, type=int, help='Port for web service')
parser_web.add_argument('--nonce', default=os.getenv('MT_WEB_NONCE', ''), type=str, help='Nonce for securing internal web server communication')

# Web client mode
parser_web_client = subparsers.add_parser('web_client', help='Run in web client mode')
parser_web_client.add_argument('--host', default='127.0.0.1', type=str, help='Host for web client')
parser_web_client.add_argument('--port', default=5003, type=int, help='Port for web client')
parser_web_client.add_argument('--nonce', default=os.getenv('MT_WEB_NONCE', ''), type=str, help='Nonce for securing internal web client communication')

# WebSocket mode
parser_ws = subparsers.add_parser('ws', help='Run in WebSocket mode')
parser_ws.add_argument('--host', default='127.0.0.1', type=str, help='Host for WebSocket service')
parser_ws.add_argument('--port', default=5003, type=int, help='Port for WebSocket service')
parser_ws.add_argument('--nonce', default=os.getenv('MT_WEB_NONCE', ''), type=str, help='Nonce for securing internal WebSocket communication')
parser_ws.add_argument('--ws-url', default='ws://localhost:5000', type=str, help='Server URL for WebSocket mode')

# API mode
parser_api = subparsers.add_parser('api', help='Run in API mode')
parser_api.add_argument('--host', default='127.0.0.1', type=str, help='Host for API service')
parser_api.add_argument('--port', default=5003, type=int, help='Port for API service')
parser_api.add_argument('--nonce', default=os.getenv('MT_WEB_NONCE', ''), type=str, help='Nonce for securing internal API server communication')


# Generares dict with a default value for each argument
DEFAULT_ARGS = vars(parser.parse_args([]))
