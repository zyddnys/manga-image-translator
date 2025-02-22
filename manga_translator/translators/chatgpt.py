import re
import asyncio
import time
from typing import List, Dict

from .config_gpt import ConfigGPT
from .common import CommonTranslator, MissingAPIKeyException, VALID_LANGUAGES
from .keys import OPENAI_API_KEY, OPENAI_HTTP_PROXY, OPENAI_API_BASE, OPENAI_MODEL

try:
    import openai
except ImportError:
    openai = None

class OpenAITranslator(ConfigGPT, CommonTranslator):
    _LANGUAGE_CODE_MAP = VALID_LANGUAGES
    _MAX_REQUESTS_PER_MINUTE = 200
    _TIMEOUT = 40
    _RETRY_ATTEMPTS = 3
    _MAX_TOKENS = 8192


    def __init__(self, check_openai_key=True):
        _CONFIG_KEY = 'chatgpt.' + OPENAI_MODEL
        ConfigGPT.__init__(self, config_key=_CONFIG_KEY)
        CommonTranslator.__init__(self)
        
        if not OPENAI_API_KEY and check_openai_key:
            raise MissingAPIKeyException('OPENAI_API_KEY environment variable required')

        client_args = {
            "api_key": OPENAI_API_KEY,
            "base_url": OPENAI_API_BASE
        }
        
        if OPENAI_HTTP_PROXY:
            from httpx import AsyncClient
            client_args["http_client"] = AsyncClient(proxies={
                "all://*openai.com": f"http://{OPENAI_HTTP_PROXY}"
            })

        self.client = openai.AsyncOpenAI(**client_args)
        self.token_count = 0
        self.token_count_last = 0
        self._last_request_ts = 0


    def parse_args(self, args: CommonTranslator):
        self.config = args.chatgpt_config


    def _cannot_assist(self, response: str) -> bool:
        # Common refusal terms
        ERROR_KEYWORDS = [
            # ENG_KEYWORDS
            r"I must decline",
            r'(i(\'m| am)?\s+)?sorry(.|\n)*?(can(\'t|not)|unable to|cannot)\s+(assist|help)',
            # CHINESE_KEYWORDS (using regex patterns)
            r"抱歉，?我(无法|不能)",  # Matches "抱歉，我无法" or "抱歉我不能"
            r"对不起，?我(无法|不能)",  # Matches "对不起，我无法" or "对不起我不能"
            r"我无法(满足|回答|处理)",  # Matches "我无法满足" or "我无法回答" or "我无法处理"
            r"这超出了我的范围",  # Matches "这超出了我的范围"
            r"我不便回答",  # Matches "我不便回答"
            r"我不能提供相关建议",  # Matches "我不能提供相关建议"
            r"这类内容我不能处理",  # Matches "这类内容我不能处理"
            r"我需要婉拒",  # Matches "我需要婉拒"
            # JAPANESE_KEYWORDS
            r"申し訳ありませんが",
        ]

        # Use regex to check for common variants of refusal phrases.
        #   Check for `ERROR_KEYWORDS` for other variants, languages
        refusal_pattern = re.compile(
            '|'.join(ERROR_KEYWORDS),re.IGNORECASE
        )

        # Check if any refusal pattern matches the response
        return bool(refusal_pattern.search(response.strip().lower()))

    def _assemble_prompts(self, from_lang: str, to_lang: str, queries: List[str]):
        prompt=''

        if self.include_template:
            prompt = self.prompt_template.format(to_lang=to_lang)
        
        for i, query in enumerate(queries):
            prompt += f"\n<|{i+1}|>{query}"
        
        return [prompt.lstrip()], len(queries)

    async def _translate(self, from_lang: str, to_lang: str, queries: List[str]) -> List[str]:
        translations = [''] * len(queries)
        prompt, _ = self._assemble_prompts(from_lang, to_lang, queries)
        
        for attempt in range(self._RETRY_ATTEMPTS):
            try:
                response_text = await self._request_translation(to_lang, prompt[0])
                translations = self._parse_response(response_text, queries)
                return translations
            except Exception as e:
                self.logger.warning(f"Translation attempt {attempt+1} failed: {str(e)}")
                if attempt == self._RETRY_ATTEMPTS - 1:
                    raise
                await asyncio.sleep(1)
        
        return translations

    def _parse_response(self, response: str, queries: List[str]) -> List[str]:
        # Initialize output list as a copy of the input
        #   Any skipped/omitted values will be filtered out as:
        #       `Translation identical to queries`
        translations = queries.copy()

        # Testing suggests ChatGPT refusals are all-or-nothing. 
        #   If partial responses do occur, this should may benefit from revising.
        if self._cannot_assist(response):
            self.logger.error(f'Refusal message detected in response. Skipping.')  
            return translations


        expected_count=len(translations)

        # Use translation ID to position value in list `translations`
        #   Parse output to grab translation ID
        #   Use translation ID to position in a list

        # Use regex to extract response 
        response=self.extract_capture_groups(response, rf"{self.rgx_capture}")

        # Extract IDs and translations from the response
        translation_matches = list(re.finditer(r'<\|(\d+)\|>(.*?)(?=(<\|\d+\|>|$))', 
                                    response, re.DOTALL)
                                )

        # Insert translations into their respective positions based on IDs:
        for match in translation_matches:
            id_num = int(match.group(1))
            translation = match.group(2).strip()
            
            # Ensure the ID is within the expected range
            if id_num < 1 or id_num > expected_count:
                raise ValueError(f"ID {id_num} in response is out of range (expected 1 to {expected_count})")
            
            # Insert the translation at the correct position
            translations[id_num - 1] = translation
        
        return translations

    async def _request_translation(self, to_lang: str, prompt: str) -> str:
        messages = [{'role': 'system', 'content': self.chat_system_template.format(to_lang=to_lang)}]
        
        if to_lang in self.chat_sample:
            messages.append({'role': 'user', 'content': self.chat_sample[to_lang][0]})
            messages.append({'role': 'assistant', 'content': self.chat_sample[to_lang][1]})
            
        messages.append({'role': 'user', 'content': prompt})

        self.logger.debug("-- GPT prompt --\n" + 
                "\n".join(f"{msg['role'].capitalize()}:\n {msg['content']}" for msg in messages) +
                "\n"
            )

        try:
            response = await self.client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages,
                max_tokens=self._MAX_TOKENS // 2,
                temperature=self.temperature,
                top_p=self.top_p,
                timeout=self._TIMEOUT
            )
            
            self.logger.debug("\n-- GPT Response --\n" +
                                response.choices[0].message.content +
                                "\n------------------\n"
                            )

            if response.usage:
                self.token_count += response.usage.total_tokens
                self.token_count_last = response.usage.total_tokens
            
            if not response.choices:
                raise ValueError("Empty response from OpenAI API")
            
            return response.choices[0].message.content

        except openai.RateLimitError as e:
            self.logger.error("Rate limit exceeded. Consider upgrading your plan or adding payment method.")
            raise
        except openai.APIError as e:
            self.logger.error(f"API error: {e.message}")
            raise
        except Exception as e:
            self.logger.error(f"Error in _request_translation: {str(e)}")
            raise

    async def _ratelimit_sleep(self):
        if self._MAX_REQUESTS_PER_MINUTE > 0:
            now = time.time()
            delay = 60 / self._MAX_REQUESTS_PER_MINUTE
            if now - self._last_request_ts < delay:
                await asyncio.sleep(delay - (now - self._last_request_ts))
            self._last_request_ts = time.time()
