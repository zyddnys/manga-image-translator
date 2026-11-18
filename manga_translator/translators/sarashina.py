import asyncio
import time
import aiohttp

from .common import CommonTranslator, InvalidServerResponse, MissingAPIKeyException
from .keys import SARASHINA_API_KEY, SARASHINA_API_BASE, SARASHINA_REQUESTS_PER_MINUTE

class SarashinaTranslator(CommonTranslator):
    # The Translate API only supports Japanese <-> English
    _LANGUAGE_CODE_MAP = {
        'JPN': 'ja',
        'ENG': 'en',
    }
    # Limit of the `text` parameter, counted in Unicode code points
    _MAX_TEXT_LENGTH = 5000
    _RATELIMIT_RETRY_ATTEMPTS = 3

    def __init__(self):
        super().__init__()
        if not SARASHINA_API_KEY:
            raise MissingAPIKeyException('Please set the SARASHINA_API_KEY environment variable before using the sarashina translator.')
        self._request_interval = 60 / max(SARASHINA_REQUESTS_PER_MINUTE, 1)
        self._last_sarashina_request_ts = 0

    async def _translate(self, from_lang, to_lang, queries):
        # source_lang is required and must differ from target_lang, and only ja/en exist
        if from_lang == 'auto':
            from_lang = 'ja' if to_lang == 'en' else 'en'
        # The API accepts a single text per request but preserves newlines in the
        # response, so batch multiple queries into one request to reduce request count.
        translations = []
        async with aiohttp.ClientSession() as session:
            for chunk in self._chunk_queries(queries):
                translations.extend(await self._translate_chunk(session, chunk, from_lang, to_lang))
        return translations

    def _chunk_queries(self, queries):
        # Group queries into chunks that stay under the text length limit when joined
        chunk, length = [], 0
        for query in queries:
            query = query.replace('\n', ' ')
            if chunk and length + 1 + len(query) > self._MAX_TEXT_LENGTH:
                yield chunk
                chunk, length = [], 0
            chunk.append(query)
            length += len(query) + (1 if len(chunk) > 1 else 0)
        if chunk:
            yield chunk

    async def _translate_chunk(self, session, chunk, from_lang, to_lang):
        if len(chunk) > 1:
            result = await self._do_request(session, '\n'.join(chunk), from_lang, to_lang)
            lines = [line.strip() for line in result.split('\n')]
            if lines and lines[-1] == '':
                lines.pop()
            if len(lines) == len(chunk):
                return lines
            self.logger.warning(f'Batched response line count ({len(lines)}) differs from chunk size ({len(chunk)}); falling back to per-query requests')
        translations = []
        for query in chunk:
            translations.append(await self._do_request(session, query, from_lang, to_lang))
        return translations

    async def _throttle(self):
        wait = self._last_sarashina_request_ts + self._request_interval - time.time()
        if wait > 0:
            self.logger.info(f'Sarashina ratelimit sleep: {wait:.2f}s')
            await asyncio.sleep(wait)
        self._last_sarashina_request_ts = time.time()

    async def _do_request(self, session, text, from_lang, to_lang):
        if len(text) > self._MAX_TEXT_LENGTH:
            self.logger.warning(f'Text exceeds {self._MAX_TEXT_LENGTH} characters and will be truncated')
            text = text[:self._MAX_TEXT_LENGTH]
        headers = {'Authorization': f'Bearer {SARASHINA_API_KEY}'}
        data = {'text': text, 'source_lang': from_lang, 'target_lang': to_lang}
        for attempt in range(self._RATELIMIT_RETRY_ATTEMPTS + 1):
            await self._throttle()
            async with session.post(f'{SARASHINA_API_BASE}/translate', json=data, headers=headers) as resp:
                if resp.status == 429:
                    if attempt >= self._RATELIMIT_RETRY_ATTEMPTS:
                        raise InvalidServerResponse('Sarashina rate limit exceeded. Lower SARASHINA_REQUESTS_PER_MINUTE.')
                    # `Retry-After` contains the seconds until the next rate limit window
                    retry_after = float(resp.headers.get('Retry-After', '60'))
                    self.logger.info(f'Sarashina rate limit exceeded; retrying in {retry_after}s')
                    self._last_sarashina_request_ts = time.time()
                    await asyncio.sleep(retry_after)
                    continue
                if resp.status != 200:
                    raise InvalidServerResponse(f'Sarashina returned HTTP {resp.status}: {await resp.text()}\nIs the API key set correctly?')
                result = await resp.json(content_type=None)
                if not isinstance(result, dict) or 'translated_text' not in result:
                    raise InvalidServerResponse(f'Sarashina returned invalid response ({resp.status}): {result}\nIs the API key set correctly?')
                return result['translated_text']
