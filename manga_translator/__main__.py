import os
import asyncio
import logging
from argparse import Namespace

from manga_translator import Config
from manga_translator.args import parser, reparse
from .manga_translator import (
    set_main_logger, load_dictionary, apply_dictionary,
)
from .args import parser
from .utils import (
    BASE_PATH,
    init_logging,
    get_logger,
    set_log_level,
    natural_sort,
)

# TODO: Dynamic imports to reduce ram usage in web(-server) mode. Will require dealing with args.py imports.

async def dispatch(args: Namespace):
    args_dict = vars(args)

    logger.info(f'Running in {args.mode} mode')

    if args.mode == 'local':
        if not args.input:
            raise Exception('No input image was supplied. Use -i <image_path>')
        from manga_translator.mode.local import MangaTranslatorLocal
        translator = MangaTranslatorLocal(args_dict)

        # Load pre-translation and post-translation dictionaries
        pre_dict = load_dictionary(args.pre_dict)
        post_dict = load_dictionary(args.post_dict)

        if len(args.input) == 1 and os.path.isfile(args.input[0]):
            dest = os.path.join(BASE_PATH, 'result/final.png')
            args.overwrite = True # Do overwrite result/final.png file

            # Apply pre-translation dictionaries
            await translator.translate_path(args.input[0], dest, args_dict)
            for textline in translator.textlines:
                textline.text = apply_dictionary(textline.text, pre_dict)
                logger.info(f'Pre-translation dictionary applied: {textline.text}')

            # Apply post-translation dictionaries
            for textline in translator.textlines:
                textline.translation = apply_dictionary(textline.translation, post_dict)
                logger.info(f'Post-translation dictionary applied: {textline.translation}')

        else: # batch
            dest = args.dest
            for path in natural_sort(args.input):
                    # Apply pre-translation dictionaries
                await translator.translate_path(path, dest, args_dict)
                for textline in translator.textlines:
                    textline.text = apply_dictionary(textline.text, pre_dict)
                    logger.info(f'Pre-translation dictionary applied: {textline.text}')

                    # Apply post-translation dictionaries
                for textline in translator.textlines:
                    textline.translation = apply_dictionary(textline.translation, post_dict)
                    logger.info(f'Post-translation dictionary applied: {textline.translation}')

    elif args.mode == 'ws':
        from manga_translator.mode.ws import MangaTranslatorWS
        translator = MangaTranslatorWS(args_dict)
        await translator.listen(args_dict)

    elif args.mode == 'shared':
        from manga_translator.mode.share import MangaShare
        translator = MangaShare(args_dict)
        await translator.listen(args_dict)
    elif args.mode == 'config-help':
        import json
        config = Config.schema()
        print(json.dumps(config, indent=2))



if __name__ == '__main__':
    args = None
    init_logging()
    try:
        args, unknown = parser.parse_known_args()
        args = Namespace(**{**vars(args), **vars(reparse(unknown))})
        set_log_level(level=logging.DEBUG if args.verbose else logging.INFO)
        logger = get_logger(args.mode)
        set_main_logger(logger)
        if args.mode != 'web':
            logger.debug(args)

        asyncio.run(dispatch(args))
    except KeyboardInterrupt:
        if not args or args.mode != 'web':
            print()
    except Exception as e:
        logger.error(f'{e.__class__.__name__}: {e}',
                     exc_info=e if args and args.verbose else None)
