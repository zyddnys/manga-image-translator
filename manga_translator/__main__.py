import asyncio
import os
import logging
from argparse import Namespace

from .manga_translator import MangaTranslator, MangaTranslatorWeb, MangaTranslatorWS, set_logger
from .args import parser
from .utils import BASE_PATH

async def dispatch(args: Namespace):
    args_dict = vars(args)

    logger.info(f'Running in {args.mode} mode')

    if args.mode in ('demo', 'batch'):
        if not args.input:
            raise Exception('No input image was supplied. Use -i <image_path>')
        if args.mode == 'demo':
            if not os.path.isfile(args.input):
                raise FileNotFoundError(f'Invalid image file path for demo mode: "{args.input}"')
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
        translator = MangaTranslatorWS(args_dict)
        await translator.listen(args_dict)

if __name__ == '__main__':
    args = None
    try:
        args = parser.parse_args()
        logging.root.setLevel(level=logging.DEBUG if args.verbose else logging.INFO)
        logger = logging.getLogger(args.mode)
        logger.debug(args)
        set_logger(logger)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(dispatch(args))
    except KeyboardInterrupt:
        print()
    except Exception as e:
        logger.error(f'{e.__class__.__name__}: {e}',
                     exc_info=e if args and args.verbose else None)
