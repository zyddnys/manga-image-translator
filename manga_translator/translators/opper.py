import asyncio
import time
from typing import List

try:
    import openai
except ImportError:
    openai = None

from .common import MissingAPIKeyException
from .common_gpt import CommonGPTTranslator
from .keys import OPPER_API_KEY, OPPER_API_BASE, OPPER_MODEL


class OpperTranslator(CommonGPTTranslator):
    """Opper (https://opper.ai), an EU-hosted OpenAI-compatible gateway.

    One endpoint serves models from many providers, so the model is chosen
    with ``OPPER_MODEL`` rather than being fixed here. Everything on the
    wire is plain OpenAI chat completions, which is why this translator only
    builds the client and leaves prompt assembly, response parsing and rate
    limiting to ``CommonGPTTranslator``.
    """

    _MAX_REQUESTS_PER_MINUTE = 200
    _TIMEOUT = 40                  # seconds to wait for a response before retrying
    _RETRY_ATTEMPTS = 3            # retries for a failed request
    _TIMEOUT_RETRY_ATTEMPTS = 3    # retries for a timed-out request
    _MAX_TOKENS = 8192

    _RETURN_PROMPT = False
    _INCLUDE_TEMPLATE = False

    def __init__(self, check_opper_key=True):
        # Nest the config under the model so a gpt_config file can hold
        # per-model prompts, the way deepseek.py does.
        CommonGPTTranslator.__init__(self, config_key='opper.' + OPPER_MODEL)

        if not OPPER_API_KEY and check_opper_key:
            raise MissingAPIKeyException('OPPER_API_KEY environment variable required')

        # The key is checked first because openai>=1.66 refuses to construct a
        # client without credentials, which would raise OpenAIError before the
        # check above could give the clearer message. The placeholder keeps the
        # keyless construction path (check_opper_key=False) working, the
        # way custom_openai.py does it.
        self.client = openai.AsyncOpenAI(api_key=OPPER_API_KEY or 'opper',
                                         base_url=OPPER_API_BASE)

        self.token_count = 0
        self.token_count_last = 0
        self.config = None

    @property
    def MODEL(self) -> str:
        """The id `_assemble_request` puts in the request body."""
        return OPPER_MODEL

    def count_tokens(self, text: str) -> int:
        """One token per UTF-8 byte.

        The gateway fronts many model families with different tokenizers, so
        there is no single correct count. `CommonGPTTranslator.count_tokens`
        names this ratio as the safe upper bound to use when the true count is
        not obtainable; over-estimating only splits batches a little earlier.
        """
        return len(text.encode('utf-8'))

    def _format_prompt_log(self, to_lang: str, prompt: str) -> str:
        lines = ['System:', self.chat_system_template.format(to_lang=to_lang)]
        sample = self.get_chat_sample(to_lang)
        if sample:
            lines += ['User:', sample[0], 'Assistant:', sample[1]]
        lines += ['User:', prompt]
        return '\n'.join(lines)

    async def _translate(self, from_lang: str, to_lang: str, queries: List[str]) -> List[str]:
        translations = [''] * len(queries)
        self.logger.debug(f'Temperature: {self.temperature}, TopP: {self.top_p}')

        async def translate_batch(batch_queries, batch_indices, split_level=0):
            split_prefix = ' (split)' if split_level > 0 else ''
            prompt, query_size = self._assemble_prompts(from_lang, to_lang, batch_queries).__next__()
            self.logger.debug(f'-- Opper Prompt{split_prefix} --\n'
                              + self._format_prompt_log(to_lang, prompt))

            for attempt in range(self._RETRY_ATTEMPTS):
                try:
                    response = await self._request_with_timeout(to_lang, prompt)
                    self.logger.debug(f'-- Opper Response{split_prefix} --\n' + response)

                    new_translations = self._parse_response(response, batch_queries)
                    if len(new_translations) < query_size:
                        self.logger.warning(
                            f'Incomplete response, {self._RETRY_ATTEMPTS - attempt - 1} attempt(s) left '
                            'before splitting the batch.')
                        continue

                    new_translations = new_translations[:query_size]
                    if any(not t.strip() for t in new_translations):
                        self.logger.warning('Empty translations detected. Resplitting the batch.')
                        break

                    for index, translation in zip(batch_indices, new_translations):
                        translations[index] = translation
                    self.logger.info(
                        f'Batch translated: {len([t for t in translations if t])}/{len(queries)} completed.')
                    return True

                except openai.RateLimitError:
                    self.logger.warning('Rate limited by Opper. Retrying.')
                    await asyncio.sleep(1)
                except openai.APIError as error:
                    if attempt == self._RETRY_ATTEMPTS - 1:
                        self.logger.error('Opper returned a server error. Use a different translator '
                                          'or try again later.')
                        raise
                    self.logger.warning(f'Restarting request after a server error: {error}')
                    await asyncio.sleep(1)
                except Warning as warning:
                    # `_parse_response` raises this when a single-query response
                    # came back without its <|1|> prefix.
                    self.logger.warning(f'{warning} Retrying. (Attempt {attempt + 1})')
                except Exception as error:
                    self.logger.error(f'Error during translation attempt: {error}')
                    if attempt == self._RETRY_ATTEMPTS - 1:
                        raise
                    await asyncio.sleep(1)

            # Retries exhausted: halve the batch and try each half on its own.
            if split_level >= 5:
                self.logger.error('Maximum split attempts reached. Unable to translate:')
                for index in batch_indices:
                    self.logger.error(f'Query: {queries[index]}')
                return False

            self.logger.warning('Retry limit reached. Splitting the translation batch.')
            mid = len(batch_queries) // 2
            halves = [(batch_queries[:mid], batch_indices[:mid]),
                      (batch_queries[mid:], batch_indices[mid:])]
            results = await asyncio.gather(
                *(translate_batch(q, i, split_level + 1) for q, i in halves if q))
            return all(results)

        await translate_batch(queries, list(range(len(queries))))

        if self.token_count_last:
            self.logger.info(f'Used {self.token_count_last} tokens (Total: {self.token_count})')
        return translations

    async def _request_with_timeout(self, to_lang: str, prompt: str) -> str:
        """Run one request, restarting it while it overruns `_TIMEOUT`."""
        for timeout_attempt in range(self._TIMEOUT_RETRY_ATTEMPTS + 1):
            task = asyncio.create_task(self._request_translation(to_lang, prompt))
            started = time.time()
            while not task.done():
                await asyncio.sleep(0.1)
                if time.time() - started > self._TIMEOUT + (timeout_attempt * self._TIMEOUT / 2):
                    task.cancel()
                    break
            else:
                return await task
            self.logger.warning(f'Restarting request due to timeout. Attempt: {timeout_attempt + 1}')
        raise Exception('Opper did not respond quickly enough.')

    async def _request_translation(self, to_lang: str, prompt: str) -> str:
        await self._ratelimit_sleep()
        response = await self.client.chat.completions.create(**self._assemble_request(to_lang, prompt))

        usage = getattr(response, 'usage', None)
        self.token_count_last = getattr(usage, 'total_tokens', 0) or 0
        self.token_count += self.token_count_last

        return response.choices[0].message.content or ''
