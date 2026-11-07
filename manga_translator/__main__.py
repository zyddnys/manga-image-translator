import os
import sys
import asyncio
import logging
import tkinter as tk
import threading
import queue
import copy
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
    spawn_detached_editor_ui,
)


    
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

        raw_input = args.input if isinstance(args.input, list) and args.input else args.input
        
        if raw_input and os.path.isfile(str(raw_input)) and (not isinstance(args.input, list) or len(args.input) == 1):
            dest = os.path.join(BASE_PATH, 'result/final.png')
            args.overwrite = True

            # Apply pre-translation dictionaries
            await translator.translate_path(str(raw_input), dest, args_dict)
            for textline in translator.textlines:
                textline.text = apply_dictionary(textline.text, pre_dict)
                logger.info(f'Pre-translation dictionary applied: {textline.text}')

            # Apply post-translation dictionaries
            for textline in translator.textlines:
                textline.translation = apply_dictionary(textline.translation, post_dict)
                logger.info(f'Post-translation dictionary applied: {textline.translation}')

        else: # batch
            dest = args.dest
            target_iterable = args.input if isinstance(args.input, list) else [args.input]
            for path in natural_sort(target_iterable):
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
    ui_thread = None
    init_logging()
    #os.environ['PYTORCH_CUDA_ALLOC_CONF']="expandable_segments:True"
    try:
        args, unknown = parser.parse_known_args()
        args = Namespace(**{**vars(args), **vars(reparse(unknown))})
        set_log_level(level=logging.DEBUG if args.verbose else logging.INFO)
        logger = get_logger(args.mode)
        set_main_logger(logger)
        if args.mode != 'web':
            logger.debug(args)
        
        if getattr(args, 'multi_gpu', False) or '--multi-gpu' in sys.argv:
            ui_thread = threading.Thread(
                target=spawn_detached_editor_ui,
                args=(None,),
                daemon=False
            )
            ui_thread.start()
            # Route processing out to our helper script located cleanly inside utils module layers
            from .utils.multi_gpu import handle_multi_gpu_execution
            handle_multi_gpu_execution(args, logger) 
        else:        
            if getattr(args, 'live_editor', False) or '--live-editor' in sys.argv:
                ui_thread = threading.Thread(
                    target=spawn_detached_editor_ui,
                    args=(None,),
                    daemon=False #todo True
                )
                ui_thread.start()  
            # Base execution path fallback
            asyncio.run(dispatch(args))


    except KeyboardInterrupt:
        print('\nTranslation cancelled by user.')
        sys.exit(0)
    except asyncio.CancelledError:
        print('\nTranslation cancelled by user.')
        sys.exit(0)
    except Exception as e:
        logger.error(f'{e.__class__.__name__}: {e}',
                     exc_info=e if args and args.verbose else None)

    #while not shutdown_event.is_set():
    #    time.sleep(1)
    if ui_thread is not None:
        ui_thread.join()
