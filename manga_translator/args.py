import argparse
import os

from .detection import DETECTORS
from .ocr import OCRS
from .inpainting import INPAINTERS
from .translators import VALID_LANGUAGES, TRANSLATORS

def path(string):
    if not string:
        return ''
    string = os.path.expanduser(string)
    if not os.path.exists(string):
        raise FileNotFoundError(string)
    return string

def file_path(string):
    if not string:
        return ''
    string = os.path.expanduser(string)
    if not os.path.exists(string):
        raise FileNotFoundError(string)
    return string

def dir_path(string):
    if not string:
        return ''
    string = os.path.expanduser(string)
    if not os.path.exists(string):
        raise FileNotFoundError(string)
    return string

parser = argparse.ArgumentParser(description='Seamlessly translate mangas into a chosen language')
parser.add_argument('-m', '--mode', default='demo', type=str, choices=['demo', 'batch', 'web', 'web2', 'ws'], help='Run demo in either single image demo mode (demo), web service mode (web) or batch translation mode (batch)')
parser.add_argument('-i', '--input', default='', type=path, help='Path to an image file if using demo mode, or path to an image folder if using batch mode')
parser.add_argument('-o', '--dest', default='', type=str, help='Path to the destination folder for translated images in batch mode')
parser.add_argument('-l', '--target-lang', default='CHS', type=str, choices=VALID_LANGUAGES, help='Destination language')
parser.add_argument('-v', '--verbose', action='store_true', help='Print debug info and save intermediate images in result folder')
parser.add_argument('--detector', default='default', type=str, choices=DETECTORS, help='Text detector used for creating a text mask from an image')
parser.add_argument('--ocr', default='48px_ctc', type=str, choices=OCRS, help='Optical character recognition (OCR) model to use')
parser.add_argument('--inpainter', default='lama_mpe', type=str, choices=INPAINTERS, help='Inpainting model to use')
parser.add_argument('--translator', default='google', type=str, choices=TRANSLATORS, help='Language translator to use')
parser.add_argument('--mtpe', action='store_true', help='Turn on/off machine translation post editing (MTPE) on the command line (works only on linux right now)')

g = parser.add_mutually_exclusive_group()
g.add_argument('--use-cuda', action='store_true', help='Turn on/off cuda')
g.add_argument('--use-cuda-limited', action='store_true', help='Turn on/off cuda (excluding offline translator)')
parser.add_argument_group(g)

parser.add_argument('--detection-size', default=1536, type=int, help='Size of image used for detection')
parser.add_argument('--det-rearrange-max-batches', default=4, type=int, help='Max batch size produced by the rearrangement of image with extreme aspectio, reduce it if cuda OOM')
parser.add_argument('--inpainting-size', default=2048, type=int, help='Size of image used for inpainting (too large will result in OOM)')
parser.add_argument('--unclip-ratio', default=2.3, type=float, help='How much to extend text skeleton to form bounding box')
parser.add_argument('--box-threshold', default=0.7, type=float, help='Threshold for bbox generation')
parser.add_argument('--text-threshold', default=0.5, type=float, help='Threshold for text detection')
parser.add_argument('--text-mag-ratio', default=1, type=int, help='Text rendering magnification ratio, larger means higher quality')
parser.add_argument('--font-size-offset', default=0, type=int, help='Offset font size by a given amount, positive number increase font size and vice versa')

g = parser.add_mutually_exclusive_group()
g.add_argument('--force-horizontal', action='store_true', help='Force text to be rendered horizontally')
g.add_argument('--force-vertical', action='store_true', help='Force text to be rendered vertically')
parser.add_argument_group(g)

parser.add_argument('--upscale-ratio', default=None, type=int, choices=[1, 2, 4, 8, 16, 32], help='waifu2x image upscale ratio')
parser.add_argument('--manga2eng', action='store_true', help='Render english text translated from manga with some typesetting')
parser.add_argument('--font-path', default='', type=file_path, help='Path to font file')
parser.add_argument('--host', default='127.0.0.1', type=str, help='Used by web module to decide which host to attach to')
parser.add_argument('--port', default=5003, type=int, help='Used by web module to decide which port to attach to')
parser.add_argument('--log-web', action='store_true', help='Used by web module to decide if web logs should be surfaced')
parser.add_argument('--ws-url', default='ws://localhost:5000', type=str, help='Server URL for WebSocket mode')

# Generares dict with a default value for each argument
DEFAULT_ARGS = vars(parser.parse_args([]))
