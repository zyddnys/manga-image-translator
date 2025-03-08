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

    # ---- 关键参数 ----
    _MAX_REQUESTS_PER_MINUTE = 200
    _TIMEOUT = 30                # 每次请求的超时时间
    _RETRY_ATTEMPTS = 2          # 对同一个批次的最大整体重试次数
    _TIMEOUT_RETRY_ATTEMPTS = 3  # 请求因超时被取消后，最大尝试次数
    _RATELIMIT_RETRY_ATTEMPTS = 3# 遇到 429 等限流时的最大尝试次数
    _MAX_SPLIT_ATTEMPTS = 3      # 递归拆分批次的最大层数
    _MAX_TOKENS = 8192           # prompt+completion 的最大 token (可按模型类型调整)

    # 原脚本里的关键模板或分割标记
    _ERROR_KEYWORDS = [
        # ENG_KEYWORDS
        r"I must decline",
        r'(i(\'m| am)?\s+)?sorry(.|\n)*?(can(\'t|not)|unable to|cannot)\s+(assist|help)',
        # CHINESE_KEYWORDS (using regex patterns)
        r"抱歉，?我(无法[将把]?|不[能会]?)", 
        r"对不起，?我(无法[将把]?|不[能会]?)",  
        r"我无法(满足|回答|处理|提供)",  
        r"这超出了我的范围",  
        r"我不便回答",  
        r"我不能提供相关建议",  
        r"这类内容我不能处理",  
        r"我需要婉拒", 
        r"翻译或生成", #deepseek高频
        # JAPANESE_KEYWORDS
        r"申し訳ありませんが",    
        
    ]

    def __init__(self, check_openai_key=True):
        # ConfigGPT 的初始化
        _CONFIG_KEY = 'chatgpt.' + OPENAI_MODEL
        ConfigGPT.__init__(self, config_key=_CONFIG_KEY)
        CommonTranslator.__init__(self)

        if not OPENAI_API_KEY and check_openai_key:
            raise MissingAPIKeyException('OPENAI_API_KEY environment variable required')

        # 根据代理与基础URL等参数实例化 openai.AsyncOpenAI 客户端
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
        """如果你有外部参数要解析，可在此对 self.config 做更新"""
        self.config = args.chatgpt_config

    def _cannot_assist(self, response: str) -> bool:
        """
        判断是否出现了常见的 "我不能帮你" / "我拒绝" 等拒绝关键词。
        """
        resp_lower = response.strip().lower()
        for kw in self._ERROR_KEYWORDS:
            if kw.lower() in resp_lower:
                return True
        return False

    async def _ratelimit_sleep(self):
        """
        在请求前先做一次简单的节流 (如果 _MAX_REQUESTS_PER_MINUTE > 0)。
        """
        if self._MAX_REQUESTS_PER_MINUTE > 0:
            now = time.time()
            delay = 60.0 / self._MAX_REQUESTS_PER_MINUTE
            elapsed = now - self._last_request_ts
            if elapsed < delay:
                await asyncio.sleep(delay - elapsed)
            self._last_request_ts = time.time()

    def _assemble_prompts(self, from_lang: str, to_lang: str, queries: List[str]):
        """
        原脚本中用来把多个 query 组装到一个 Prompt。
        同时可以做长度控制，如果过长就切分成多个 prompt。
        这里演示一个简单的 chunk 逻辑：
          - 根据字符长度 roughly 判断
          - 也可以用更准确的 tokens 估算
        ps.实际没啥用
        """
        MAX_CHAR_PER_PROMPT = self._MAX_TOKENS * 4  # 粗略: 1 token ~ 4 chars
        chunk_queries = []
        current_length = 0
        batch = []

        for q in queries:
            # +10 给一些余量，比如加上 <|1|> 的标记等
            if current_length + len(q) + 10 > MAX_CHAR_PER_PROMPT and batch:
                # 输出当前 batch
                chunk_queries.append(batch)
                batch = []
                current_length = 0
            batch.append(q)
            current_length += len(q) + 10
        if batch:
            chunk_queries.append(batch)

        # 逐个批次生成 prompt
        for this_batch in chunk_queries:
            prompt = ""
            if self.include_template:
                prompt = self.prompt_template.format(to_lang=to_lang)
            # 加上分行内容
            for i, query in enumerate(this_batch):
                prompt += f"\n<|{i+1}|>{query}"
            yield prompt.lstrip(), len(this_batch)

    async def _translate(self, from_lang: str, to_lang: str, queries: List[str]) -> List[str]:
        """
        核心翻译逻辑：
            1. 把 queries 拆成多个 prompt 批次
            2. 对每个批次调用 translate_batch，并将结果写回 translations
        """
        translations = [''] * len(queries)
        # 记录当前处理到 queries 列表的哪个位置
        idx_offset = 0

        # 分批处理
        for prompt, batch_size in self._assemble_prompts(from_lang, to_lang, queries):
            # 实际要翻译的子列表
            batch_queries = queries[idx_offset : idx_offset + batch_size]
            indices = list(range(idx_offset, idx_offset + batch_size))

            # 执行翻译
            success, partial_results = await self._translate_batch(
                from_lang, to_lang, batch_queries, indices, prompt, split_level=0
            )
            # 将结果写入 translations
            for i, r in zip(indices, partial_results):
                translations[i] = r

            idx_offset += batch_size

        return translations

    async def _translate_batch(
        self,
        from_lang: str,
        to_lang: str,
        batch_queries: List[str],
        batch_indices: List[int],
        prompt: str,
        split_level: int = 0
    ):
        """
        尝试翻译 batch_queries。若失败或返回不完整，则进一步拆分。
        :return: (bool 是否成功, List[str] 对应每个 query 的翻译结果)
        """
        partial_results = [''] * len(batch_queries)
        # 如果没有查询就直接返回
        if not batch_queries:
            return True, partial_results

        # 进行 _RETRY_ATTEMPTS 次重试
        for attempt in range(self._RETRY_ATTEMPTS):
            try:
                # 1) 发起请求
                response_text = await self._request_with_retry(to_lang, prompt)
                
                # 2) 解析 response
                #    直接在这里进行解析 + 校验，不通过则抛异常
                new_translations = re.split(r'<\|\d+\|>', response_text)
                
                # 删除正则分割后产生的第一个空串
                if not new_translations[0].strip():
                    new_translations = new_translations[1:]              

                # 检查风控词，这是整体检测，需要前置
                if self._cannot_assist(response_text):
                    self.logger.warning(f"Detected refusal message from model. Will retry (attempt {attempt+1}).")
                    continue

                # 处理query只有1，返回内容也是1但是没有前缀的情况。这往往是错误返回，例如模型可能返回翻译无意义的解释说明。
                if len(batch_queries) == 1 and len(new_translations) == 1 and not re.match(r'^\s*<\|1\|>', response_text):
                    self.logger.warning(f'Single query response does not contain prefix, retrying...(Attempt {attempt + 1})')
                    continue
                
                # 如果返回个数小于本批数量，可能需要改用别的拆分方式(比如按行切)
                if len(new_translations) < len(batch_queries):
                    # 这里演示，简单再按行分隔
                    alt_splits = response_text.splitlines()
                    if len(alt_splits) == len(batch_queries):
                        new_translations = alt_splits
                    if len(alt_splits) > len(batch_queries):
                        continue

                # 检查数量，若依旧不足则说明不完整
                if len(new_translations) < len(batch_queries):
                    self.logger.warning(
                        f"[Attempt {attempt+1}] Batch response is incomplete. "
                        f"Expect {len(batch_queries)}, got {len(new_translations)}"
                    )
                    # 继续下一次重试
                    continue
                
                # 去除多余空行、前后空格
                new_translations = [t.strip() for t in new_translations]

                # 判断是否有明显的空翻译(检测到1个空串就报错)
                if any(not t for t in new_translations):
                    self.logger.warning(
                        f"[Attempt {attempt+1}] Empty translation detected. Retrying..."
                    )
                    # 需要注意，此处也可换成break直接进入分割逻辑。原因是若出现空结果时，不断重试出现正确结果的效率相对较低，可能直到用尽重试错误依然无解。但是为了尽可能确保翻译质量，使用了continue，并相应地下调重试次数以抵消影响。
                    continue

                # 一切正常，写入 partial_results
                for i in range(len(batch_queries)):
                    partial_results[i] = new_translations[i]

                # 成功
                self.logger.info(
                    f"Batch of size {len(batch_queries)} translated OK at attempt {attempt+1} (split_level={split_level})."
                )
                return True, partial_results

            except Exception as e:
                self.logger.warning(
                    f"Batch translate attempt {attempt+1} failed with error: {str(e)}"
                )
                if attempt < self._RETRY_ATTEMPTS - 1:
                    await asyncio.sleep(1)
                else:
                    self.logger.warning("Max attempts reached, will try to split if possible.")

        # 如果代码能执行到这里，说明前面多个重试都失败或不完整 => 尝试拆分。通过减小每次请求的文本量，或者隔离可能导致问题(如产生空行、风控词)的特定 query，来尝试解决问题
        if split_level < self._MAX_SPLIT_ATTEMPTS and len(batch_queries) > 1:
            self.logger.warning(
                f"Splitting batch of size {len(batch_queries)} at split_level={split_level}"
            )
            mid = len(batch_queries) // 2
            left_queries = batch_queries[:mid]
            right_queries = batch_queries[mid:]

            left_indices = batch_indices[:mid]
            right_indices = batch_indices[mid:]

            # 递归翻译左半部分
            left_prompt, _ = next(self._assemble_prompts(from_lang, to_lang, left_queries))
            left_success, left_results = await self._translate_batch(
                from_lang, to_lang, left_queries, left_indices, left_prompt, split_level+1
            )

            # 递归翻译右半部分
            right_prompt, _ = next(self._assemble_prompts(from_lang, to_lang, right_queries))
            right_success, right_results = await self._translate_batch(
                from_lang, to_lang, right_queries, right_indices, right_prompt, split_level+1
            )

            # 合并
            return (left_success and right_success), (left_results + right_results)
        else:
            # 不能再拆分了就返回 区分没有前缀的和分割后依然失败的
            if len(batch_queries) == 1 and not re.match(r'^\s*<\|1\|>', response_text):
                self.logger.error(
                    f"Single query translation failed after max retries due to missing prefix. size={len(batch_queries)}"
                )
            else:
                self.logger.error(
                    f"Translation failed after max retries and splits. Returning original queries. size={len(batch_queries)}"
                )
            # 失败的query全部保留
            for i in range(len(batch_queries)): 
                partial_results[i] = batch_queries[i]     
                
            return False, partial_results

    async def _request_with_retry(self, to_lang: str, prompt: str) -> str:
        """
        结合重试、超时、限流处理的请求入口。
        """
        # 这里演示3层重试: 
        #   1) 如果请求超时 => 重新发起(最多 _TIMEOUT_RETRY_ATTEMPTS 次)
        #   2) 如果返回 429 => 也做重试(最多 _RATELIMIT_RETRY_ATTEMPTS 次)
        #   3) 其他错误 => 重试 _RETRY_ATTEMPTS 次
        # 最终失败则抛异常
        # 也可以将下面逻辑整合到 _translate_batch 里，但保持一次请求一次处理也行。

        timeout_attempt = 0
        ratelimit_attempt = 0
        server_error_attempt = 0

        while True:
            await self._ratelimit_sleep()
            started = time.time()
            req_task = asyncio.create_task(self._request_translation(to_lang, prompt))

            try:
                # 等待请求
                while not req_task.done():
                    await asyncio.sleep(0.1)
                    if time.time() - started > self._TIMEOUT:
                        # 超时 => 取消请求并重试
                        timeout_attempt += 1
                        if timeout_attempt > self._TIMEOUT_RETRY_ATTEMPTS:
                            raise TimeoutError(
                                f"OpenAI request timed out after {self._TIMEOUT_RETRY_ATTEMPTS} attempts."
                            )
                        self.logger.warning(f"Request timed out, retrying... (attempt={timeout_attempt})")
                        req_task.cancel()
                        break
                else:
                    # 如果正常完成了
                    return req_task.result()

            except openai.RateLimitError:
                # 限流 => 重试
                ratelimit_attempt += 1
                if ratelimit_attempt > self._RATELIMIT_RETRY_ATTEMPTS:
                    raise
                self.logger.warning(f"Hit RateLimit, retrying... (attempt={ratelimit_attempt})")
                await asyncio.sleep(2)

            except openai.APIError as e:
                # 服务器错误 => 重试
                server_error_attempt += 1
                if server_error_attempt > self._RETRY_ATTEMPTS:
                    self.logger.error("Server error, giving up after several attempts.")
                    raise
                self.logger.warning(f"Server error: {str(e)}. Retrying... (attempt={server_error_attempt})")
                await asyncio.sleep(1)

            except Exception as e:
                self.logger.error(f"Unexpected error in _request_with_retry: {str(e)}")
                raise

    async def _request_translation(self, to_lang: str, prompt: str) -> str:
        """
        实际调用 openai.ChatCompletion 的请求部分。
        """
        # 构建 messages
        messages = [
            {'role': 'system', 'content': self.chat_system_template.format(to_lang=to_lang)},
        ]
        # 如果需要先给出示例对话
        if to_lang in self.chat_sample:
            messages.append({'role': 'user', 'content': self.chat_sample[to_lang][0]})
            messages.append({'role': 'assistant', 'content': self.chat_sample[to_lang][1]})

        # 最终用户请求
        messages.append({'role': 'user', 'content': prompt})

        # 打印或记录 prompt 以方便 debug
        self.logger.debug(
            "-- GPT prompt --\n"
            + "\n".join(f"{m['role'].upper()}:\n{m['content']}" for m in messages)
            + "\n----------------\n"
        )

        # 发起请求
        response = await self.client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            max_tokens=self._MAX_TOKENS // 2,
            temperature=self.temperature,
            top_p=self.top_p,
            timeout=self._TIMEOUT
        )

        if not response.choices:
            raise ValueError("Empty response from OpenAI API")

        raw_text = response.choices[0].message.content

        # 去除 <think>...</think> 标签及内容。由于某些中转api的模型的思考过程是被强制输出的，并不包含在reasoning_content中，需要额外过滤
        raw_text = re.sub(r'<think>.*?</think>', '', raw_text, flags=re.DOTALL)

        # 删除多余的空行
        cleaned_text = re.sub(r'\n\s*\n', '\n', raw_text).strip()

        # 记录 token 消耗
        if not hasattr(response, 'usage') or not hasattr(response.usage, 'total_tokens'):
            self.logger.warning("Response does not contain usage information") #第三方逆向中转api不返回token数
            self.token_count_last = 0
        else:
            self.token_count += response.usage.total_tokens
            self.token_count_last = response.usage.total_tokens

        self.logger.debug("-- GPT Response --\n" + cleaned_text + "\n----------------\n")
        return cleaned_text