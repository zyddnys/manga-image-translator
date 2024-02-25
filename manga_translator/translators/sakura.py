import re
from venv import logger

try:
    import openai
    import openai.error
except ImportError:
    openai = None
import asyncio
import time
from typing import List, Dict

from .common import CommonTranslator
from .keys import SAKURA_API_BASE


class Sakura13BTranslator(CommonTranslator):

    _TIMEOUT = 999  # Seconds to wait for a response from the server before retrying
    _RETRY_ATTEMPTS = 1  # Number of times to retry an errored request before giving up
    _TIMEOUT_RETRY_ATTEMPTS = 3  # Number of times to retry a timed out request before giving up
    _RATELIMIT_RETRY_ATTEMPTS = 3  # Number of times to retry a ratelimited request before giving up

    _CHAT_SYSTEM_TEMPLATE = (
        '你是一个轻小说翻译模型，可以流畅通顺地以日本轻小说的风格将日文翻译成简体中文，并联系上下文正确使用人称代词，不擅自添加原文中没有的代词。'
    )

    _LANGUAGE_CODE_MAP = {
        'CHS': 'Simplified Chinese',
        'JPN': 'Japanese'
    }

    def __init__(self):
        super().__init__()
        openai.api_base = SAKURA_API_BASE
        self.temperature = 0.3
        self.top_p = 0.3
        self.frequency_penalty = 0.0
        self._current_style = "normal"

    def detect_and_remove_extra_repeats(self, s: str, threshold: int = 10, remove_all=True):
        """
        检测字符串中是否有任何模式连续重复出现超过阈值，并在去除多余重复后返回新字符串。
        保留一个模式的重复。

        :param s: str - 待检测的字符串。
        :param threshold: int - 连续重复模式出现的最小次数，默认为2。
        :return: tuple - (bool, str)，第一个元素表示是否有重复，第二个元素是处理后的字符串。
        """

        repeated = False
        for pattern_length in range(1, len(s) // 2 + 1):
            i = 0
            while i < len(s) - pattern_length:
                pattern = s[i:i + pattern_length]
                count = 1
                j = i + pattern_length
                while j <= len(s) - pattern_length:
                    if s[j:j + pattern_length] == pattern:
                        count += 1
                        j += pattern_length
                    else:
                        break
                if count >= threshold:
                    repeated = True
                    # 保留一个模式的重复
                    if remove_all:
                        s = s[:i + pattern_length] + s[j:]
                    break
                i += 1
            if repeated:
                break
        return repeated, s

    def _format_prompt_log(self, prompt: str) -> str:
        return '\n'.join([
            'System:',
            self._CHAT_SYSTEM_TEMPLATE,
            'User:',
            '将下面的日文文本翻译成中文：',
            prompt,
        ])

    # str 通过/n转换为list
    def _split_text(self, text: str) -> list:
        if isinstance(text, list):
            return text
        return text.split('\n')

    def check_align(self, queries: List[str], response: str) -> bool:
        """
        检查原始文本（queries）与翻译后的文本（response）是否保持相同的行数。

        :param queries: 原始文本的列表。
        :param response: 翻译后的文本，可能是一个字符串。
        :return: 两者行数是否相同的布尔值。
        """
        # 确保response是列表形式
        translated_texts = self._split_text(response) if isinstance(response, str) else response

        # 日志记录，而不是直接打印
        print(f"原始文本行数: {len(queries)}, 翻译文本行数: {len(translated_texts)}")
        logger.warning(f"原始文本行数: {len(queries)}, 翻译文本行数: {len(translated_texts)}")

        # 检查行数是否匹配
        is_aligned = len(queries) == len(translated_texts)
        if not is_aligned:
            logger.warning("原始文本与翻译文本的行数不匹配。")

        return is_aligned

    def _delete_quotation_mark(self, texts: List[str]) -> List[str]:
        print(texts)
        new_texts = []
        for text in texts:
            text = text.strip('「」')
            new_texts.append(text)
        return new_texts

    async def _translate(self, from_lang: str, to_lang: str, queries: List[str]) -> List[str]:
        self._log_translation_details(queries)
        queries_with_markers = self._add_quotation_marks_to_queries(queries)
        response = await self._attempt_translation_with_retry(queries_with_markers)
        translations = self._process_translation_response(response, queries_with_markers)
        return self._delete_quotation_mark(translations)

    def _log_translation_details(self, queries: List[str]):
        self.logger.debug(f'Temperature: {self.temperature}, TopP: {self.top_p}')
        self.logger.debug(f'Queries: {queries}')
        text_prompt = '\n'.join(queries)
        self.logger.debug('-- Sakura Prompt --\n' + self._format_prompt_log(text_prompt) + '\n\n')

    def _add_quotation_marks_to_queries(self, queries: List[str]) -> List[str]:
        return [f'「{query}」' for query in queries]

    async def _attempt_translation_with_retry(self, queries: List[str]) -> str:
        response = await self._handle_translation_request(queries)
        if self.detect_and_remove_extra_repeats(response)[0]:
            response = await self._retry_translation_on_error(queries)
        return response

    async def _retry_translation_on_error(self, queries: List[str], error_type='model_degradation') -> str:
        if error_type == 'model_degradation':
            for _ in range(self._RETRY_ATTEMPTS):
                self._set_gpt_style("precise")
                response = await self._handle_translation_request(queries)
                if not self.detect_and_remove_extra_repeats(response)[0]:
                    return response
        # 可以添加其他错误类型的重试逻辑
        self.logger.warning(f'Retry failed for {error_type}, returning original queries.')
        return '\n'.join(queries)  # 在无法解决重复的情况下返回原始查询

    def _process_translation_response(self, response: str, queries: List[str]) -> List[str]:
        translations = self._split_text(response)
        if not self.check_align(queries, translations):
            self.logger.warning('Mismatch in the number of lines, trying to translate line by line.')
            translations = self._fallback_to_line_by_line_translation(queries)
        return translations

    async def _fallback_to_line_by_line_translation(self, queries: List[str]) -> List[str]:
        translations = []
        for query in queries:
            response = await self._handle_translation_request([query])
            translations.append(response)
        if not self.check_align(queries, translations):
            self.logger.warning('Fallback to original text due to alignment issues.')
            return queries  # 在行数不匹配的情况下回退到原始查询
        return translations

    async def _handle_translation_request(self, prompt: str) -> str:
        # 翻译请求和错误处理逻辑
        ratelimit_attempt = 0
        server_error_attempt = 0
        timeout_attempt = 0
        while True:
            request_task = asyncio.create_task(self._request_translation(prompt))
            started = time.time()
            while not request_task.done():
                await asyncio.sleep(0.1)
                if time.time() - started > self._TIMEOUT + (timeout_attempt * self._TIMEOUT / 2):
                    if timeout_attempt >= self._TIMEOUT_RETRY_ATTEMPTS:
                        raise Exception('Sakura timeout.')
                    timeout_attempt += 1
                    self.logger.warn(f'Restarting request due to timeout. Attempt: {timeout_attempt}')
                    request_task.cancel()
                    request_task = asyncio.create_task(self._request_translation(prompt))
                    started = time.time()
            try:
                response = await request_task
                break
            except openai.error.RateLimitError:
                ratelimit_attempt += 1
                if ratelimit_attempt >= self._RATELIMIT_RETRY_ATTEMPTS:
                    raise
                self.logger.warn(f'Restarting request due to ratelimiting by sakura servers. Attempt: {ratelimit_attempt}')
                await asyncio.sleep(2)
            except openai.error.APIError:
                server_error_attempt += 1
                if server_error_attempt >= self._RETRY_ATTEMPTS:
                    self.logger.error('Sakura server error. Returning original text.')
                    return prompt  # 返回原始文本而不是抛出异常
                self.logger.warn(f'Restarting request due to a server error. Attempt: {server_error_attempt}')
                await asyncio.sleep(1)
            except openai.error.APIConnectionError:
                server_error_attempt += 1
                self.logger.warn(f'Restarting request due to a server connection error. Attempt: {server_error_attempt}')
                await asyncio.sleep(1)
            except FileNotFoundError:
                self.logger.warn(f'Restarting request due to FileNotFoundError.')
                await asyncio.sleep(30)

        return response

    async def _request_translation(self, input_text_list) -> str:
        if isinstance(input_text_list, list):
            raw_text = "\n".join(input_text_list)
        else:
            raw_text = input_text_list
        extra_query = {
            'do_sample': False,
            'num_beams': 1,
            'repetition_penalty': 1.0,
        }

        response = await openai.ChatCompletion.acreate(
            model="sukinishiro",
            messages=[
                {
                    "role": "system",
                    "content": "你是一个轻小说翻译模型，可以流畅通顺地以日本轻小说的风格将日文翻译成简体中文，并联系上下文正确使用人称代词，不擅自添加原文中没有的代词。"
                },
                {
                    "role": "user",
                    "content": f"将下面的日文文本翻译成中文：{raw_text}"
                }
            ],
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=512,
            frequency_penalty=self.frequency_penalty,
            seed=-1,
            extra_query=extra_query,
        )

        # 提取并返回响应文本
        for choice in response.choices:
            if 'text' in choice:
                return choice.text

        # 如果没有找到包含文本的响应，返回第一个响应的内容（可能为空）

        return response.choices[0].message.content

    def _set_gpt_style(self, style_name: str):
        if self._current_style == style_name:
            return
        self._current_style = style_name
        if style_name == "precise":
            temperature, top_p = 0.1, 0.3
            frequency_penalty = 0.15
        elif style_name == "normal":
            temperature, top_p = 0.3, 0.3
            frequency_penalty = 0.0

        self.temperature = temperature
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
