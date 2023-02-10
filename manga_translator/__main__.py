import asyncio
import os
from argparse import Namespace

from .manga_translator import MangaTranslator, MangaTranslatorWeb, MangaTranslatorWS
from .args import parser
from .utils import BASE_PATH


async def dispatch(args: Namespace):
    args_dict = vars(args)

    # TODO: rename batch mode to normal mode
    if args.mode in ('demo', 'batch'):
        if not args.input:
            raise Exception('No input image was supplied. Use -i <image_path>')
        if args.mode == 'demo':
            dest = os.path.join(BASE_PATH, 'result/final.png')
        else:
            dest = args.dest
        translator = MangaTranslator(args_dict)
        await translator.translate_path(args.input, dest, args_dict)

    elif args.mode in ('web', 'web2'):
        translator = MangaTranslatorWeb(args_dict)
        if args.mode == 'web':
            translator.instantiate_webserver()
        await translator.listen(args_dict)

    elif args.mode == 'ws':
        translator = MangaTranslatorWS(args)

if __name__ == '__main__':
    try:
        args = parser.parse_args()
        if args.verbose:
            print(args)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(dispatch(args))
    except KeyboardInterrupt:
        print()
