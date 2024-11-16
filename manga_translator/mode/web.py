import asyncio
import time

import requests

from manga_translator import MangaTranslator, logger, Context
from manga_translator.translators import TRANSLATORS
from manga_translator.utils import add_file_logger, remove_file_logger


class MangaTranslatorWeb(MangaTranslator):
    """
    Translator client that executes tasks on behalf of the webserver in web_main.py.
    """

    def __init__(self, params: dict = None):
        super().__init__(params)
        self.host = params.get('host', '127.0.0.1')
        if self.host == '0.0.0.0':
            self.host = '127.0.0.1'
        self.port = params.get('port', 5003)
        self.nonce = params.get('nonce', '')
        self.ignore_errors = params.get('ignore_errors', True)
        self._task_id = None
        self._params = None

    async def _init_connection(self):
        available_translators = []
        from ..translators import MissingAPIKeyException, get_translator
        for key in TRANSLATORS:
            try:
                get_translator(key)
                available_translators.append(key)
            except MissingAPIKeyException:
                pass

        data = {
            'nonce': self.nonce,
            'capabilities': {
                'translators': available_translators,
            },
        }
        requests.post(f'http://{self.host}:{self.port}/connect-internal', json=data)

    async def _send_state(self, state: str, finished: bool):
        # wait for translation to be saved first (bad solution?)
        finished = finished and not state == 'finished'
        while True:
            try:
                data = {
                    'task_id': self._task_id,
                    'nonce': self.nonce,
                    'state': state,
                    'finished': finished,
                }
                requests.post(f'http://{self.host}:{self.port}/task-update-internal', json=data, timeout=20)
                break
            except Exception:
                # if translation is finished server has to know
                if finished:
                    continue
                else:
                    break

    def _get_task(self):
        try:
            rjson = requests.get(f'http://{self.host}:{self.port}/task-internal?nonce={self.nonce}',
                                 timeout=3600).json()
            return rjson.get('task_id'), rjson.get('data')
        except Exception:
            return None, None

    async def listen(self, translation_params: dict = None):
        """
        Listens for translation tasks from web server.
        """
        logger.info('Waiting for translation tasks')

        await self._init_connection()
        self.add_progress_hook(self._send_state)

        while True:
            self._task_id, self._params = self._get_task()
            if self._params and 'exit' in self._params:
                break
            if not (self._task_id and self._params):
                await asyncio.sleep(0.1)
                continue

            self.result_sub_folder = self._task_id
            logger.info(f'Processing task {self._task_id}')
            if translation_params is not None:
                # Combine default params with params chosen by webserver
                for p, default_value in translation_params.items():
                    current_value = self._params.get(p)
                    self._params[p] = current_value if current_value is not None else default_value
            if self.verbose:
                # Write log file
                log_file = self._result_path('log.txt')
                add_file_logger(log_file)

            # final.png will be renamed if format param is set
            await self.translate_path(self._result_path('input.png'), self._result_path('final.png'),
                                      params=self._params)
            print()

            if self.verbose:
                remove_file_logger(log_file)
            self._task_id = None
            self._params = None
            self.result_sub_folder = ''

    async def _run_text_translation(self, ctx: Context):
        # Run machine translation as reference for manual translation (if `--translator=none` is not set)
        text_regions = await super()._run_text_translation(ctx)

        if ctx.get('manual', False):
            logger.info('Waiting for user input from manual translation')
            requests.post(f'http://{self.host}:{self.port}/request-manual-internal', json={
                'task_id': self._task_id,
                'nonce': self.nonce,
                'texts': [r.text for r in text_regions],
                'translations': [r.translation for r in text_regions],
            }, timeout=20)

            # wait for at most 1 hour for manual translation
            wait_until = time.time() + 3600
            while time.time() < wait_until:
                ret = requests.post(f'http://{self.host}:{self.port}/get-manual-result-internal', json={
                    'task_id': self._task_id,
                    'nonce': self.nonce
                }, timeout=20).json()
                if 'result' in ret:
                    manual_translations = ret['result']
                    if isinstance(manual_translations, str):
                        if manual_translations == 'error':
                            return []
                    i = 0
                    for translation in manual_translations:
                        if not translation.strip():
                            text_regions.pop(i)
                            i = i - 1
                        else:
                            text_regions[i].translation = translation
                            text_regions[i].target_lang = ctx.translator.langs[-1]
                        i = i + 1
                    break
                elif 'cancel' in ret:
                    return 'cancel'
                await asyncio.sleep(0.1)
        return text_regions
