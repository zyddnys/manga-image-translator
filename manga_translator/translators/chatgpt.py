import re
import os
import asyncio
import time
from typing import List, Dict
from rich.console import Console  
from rich.panel import Panel

from .config_gpt import ConfigGPT
from .common import CommonTranslator, MissingAPIKeyException, VALID_LANGUAGES
from .keys import OPENAI_API_KEY, OPENAI_HTTP_PROXY, OPENAI_API_BASE, OPENAI_MODEL, OPENAI_GLOSSARY_PATH

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

    # 模型返回的风控词检测
    _ERROR_KEYWORDS = [
            # ENG_KEYWORDS
            r"I must decline",
            r'(i(\'m| am)?\s+)?sorry(.|\n)*?(can(\'t|not)|unable to|cannot)\s+(assist|help)',
            # CHINESE_KEYWORDS (using regex patterns)
            r"(抱歉，|对不起，)?我(无法[将把]|不[能会便](提供|处理)?)", 
            r"我无法(满足|回答|处理|提供)",  
            r"这超出了我的范围",   
            r"我需要婉拒", 
            r"翻译或生成", #deepseek高频
            r"[的个]内容(吧)?", #claude高频
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
        
        # 初始化术语表相关属性
        self.dict_path = OPENAI_GLOSSARY_PATH
        self.glossary_entries = {}
        if os.path.exists(self.dict_path):
            self.glossary_entries = self.load_glossary(self.dict_path)
        else:
            self.logger.warning(f"The glossary file does not exist: {self.dict_path}")

        # 添加 rich 的 Console 对象  
        self.console = Console()  

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
        集成术语表功能。
        """
        """
        The actual request part that calls openai.ChatCompletion.
        Incorporate the glossary function.
        """        
        # 构建 messages / Construct messages
        messages = [  
            {'role': 'system', 'content': self.chat_system_template.format(to_lang=to_lang)},  
        ]  

        # 提取相关术语并添加到系统消息中  / Extract relevant terms and add them to the system message
        has_glossary = False  # 添加标志表示是否有术语表 / Add a flag to indicate whether there is a glossary
        relevant_terms = self.extract_relevant_terms(prompt)  
        if relevant_terms:  
            has_glossary = True  # 设置标志 / Set the flag
            # 构建术语表字符串 / Construct the glossary string
            glossary_text = "\n".join([f"{term}->{translation}" for term, translation in relevant_terms.items()])  
            system_message = self.glossary_system_template.format(glossary_text=glossary_text)  
            messages.append({'role': 'system', 'content': system_message})  
            self.logger.info(f"Loaded {len(relevant_terms)} relevant terms from the glossary.")  
            
        # 如果需要先给出示例对话
        # Add chat samples if available
        lang_chat_samples = self.get_chat_sample(to_lang)

        # 如果需要先给出示例对话 / Provide an example dialogue first if necessary
        if hasattr(self, 'chat_sample') and lang_chat_samples:
            messages.append({'role': 'user', 'content': lang_chat_samples[0]})
            messages.append({'role': 'assistant', 'content': lang_chat_samples[1]})

        # 最终用户请求 / End-user request 
        messages.append({'role': 'user', 'content': prompt})  

        # 准备输出的 prompt 文本 / Prepare the output prompt text 
        if self.verbose_logging:  
            prompt_text = "\n".join(f"{m['role'].upper()}:\n{m['content']}" for m in messages) 
                    
            self.print_boxed(prompt_text, border_color="cyan", title="GPT Prompt")      
        else:  
            simplified_msgs = []  
            for i, m in enumerate(messages):  
                if (has_glossary and i == 1) or (i == len(messages) - 1):  
                    simplified_msgs.append(f"{m['role'].upper()}:\n{m['content']}")  
                else:  
                    simplified_msgs.append(f"{m['role'].upper()}:\n[HIDDEN CONTENT]")  
            prompt_text = "\n".join(simplified_msgs)
            # 使用 rich 输出 prompt / Use rich to output the prompt
            self.print_boxed(prompt_text, border_color="cyan", title="GPT Prompt (verbose=False)") 
        

        # 发起请求 / Initiate the request
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
        # Remove <think>...</think> tags and their contents. Since the reasoning process of some relay API models is forcibly output and not included in the reasoning_content, additional filtering is required.
        raw_text = re.sub(r'<think>.*?</think>', '', raw_text, flags=re.DOTALL)

        # 删除多余的空行 / Remove extra blank lines
        
        cleaned_text = re.sub(r'\n\s*\n', '\n', raw_text).strip()

        # 记录 token 消耗 / Record token consumption
        if not hasattr(response, 'usage') or not hasattr(response.usage, 'total_tokens'):
            self.logger.warning("Response does not contain usage information") #第三方逆向中转api不返回token数 / The third-party reverse proxy API does not return token counts
            self.token_count_last = 0
            
        # 记录 token 消耗   (rich模式) / Record token consumption (rich mode)
        # if not hasattr(response, 'usage') or not hasattr(response.usage, 'total_tokens'):  
            # warning_text = "WARNING: [OpenAITranslator] Response does not contain usage information"  
            # self.print_boxed(warning_text, border_color="yellow")  
            # self.token_count_last = 0              
            
        else:
            self.token_count += response.usage.total_tokens
            self.token_count_last = response.usage.total_tokens
        
        response_text = cleaned_text
        self.print_boxed(response_text, border_color="green", title="GPT Response")          
        return cleaned_text

      
    # ==============修改日志输出方法 (Modify Log Output Method)==============
    def print_boxed(self, text, border_color="blue", title="OpenAITranslator Output"):  
        """将文本框起来并输出到终端"""
        """Box the text and output it to the terminal"""    
        panel = Panel(text, title=title, border_style=border_color, expand=False)  
        self.console.print(panel)          
 
    # ==============以下是术语表相关函数 (Below are glossary-related functions)==============
    
    def load_glossary(self, path):
        """加载术语表文件 / Load the glossary file"""
        if not os.path.exists(path):
            self.logger.warning(f"The OpenAI glossary file does not exist: {path}")
            return {}
                
        # 检测文件类型并解析 / Detect the file type and parse it
        dict_type = self.detect_type(path)
        if dict_type == "galtransl":
            return self.load_galtransl_dic(path)
        elif dict_type == "sakura":
            return self.load_sakura_dict(path)
        elif dict_type == "mit":
            return self.load_mit_dict(path)              
        else:
            self.logger.warning(f"Unknown OpenAI glossary format: {path}")
            return {}

    def detect_type(self, dic_path):  
        """  
        检测字典类型（OpenAI专用） / Detect dictionary type (specific to OpenAI).
        """  
        with open(dic_path, encoding="utf8") as f:  
            dic_lines = f.readlines()  
        self.logger.debug(f"Detecting OpenAI dictionary type: {dic_path}")  
        if len(dic_lines) == 0:  
            return "unknown"  

        # 先判断是否为Sakura字典 / First, determine if it is a Sakura dictionary
        is_sakura = True  
        sakura_line_count = 0  
        for line in dic_lines:  
            line = line.strip()  
            if not line or line.startswith("\\\\") or line.startswith("//"):  
                continue  
                
            if "->" in line:  
                sakura_line_count += 1  
            else:  
                is_sakura = False  
                break  
        
        if is_sakura and sakura_line_count > 0:  
            return "sakura"  

        # 判断是否为Galtransl字典 / Determine if it is a Galtransl dictionary
        is_galtransl = True  
        galtransl_line_count = 0  
        for line in dic_lines:  
            line = line.strip()  
            if not line or line.startswith("\\\\") or line.startswith("//"):  
                continue  

            if "\t" in line or "    " in line:  
                galtransl_line_count += 1  
            else:  
                is_galtransl = False  
                break  
        
        if is_galtransl and galtransl_line_count > 0:  
            return "galtransl"  

        # 判断是否为MIT字典（最宽松的格式） / Determine if it is an MIT dictionary (the most lenient format)
        is_mit = True  
        mit_line_count = 0  
        for line in dic_lines:  
            line = line.strip()  
            if not line or line.startswith("#") or line.startswith("//"):  
                continue  
                
            # 排除Sakura格式特征 / Exclude Sakura format characteristics
            if "->" in line:  
                is_mit = False  
                break  
                
            # MIT格式需要能分割出源和目标两部分 / The MIT format needs to be able to split into source and target parts
            parts = line.split("\t", 1)  
            if len(parts) == 1:  # 如果没有制表符，尝试用空格分割 / If there are no tab characters, attempt to split using spaces
                parts = line.split(None, 1)  # None表示任何空白字符 / None represents any whitespace character
            
            if len(parts) >= 2:  # 确保有源和目标两部分 / Ensure there are both source and target parts
                mit_line_count += 1  
            else:  
                is_mit = False  
                break  
        
        if is_mit and mit_line_count > 0:  
            return "mit"  

        return "unknown"  

    def load_mit_dict(self, dic_path):
        """载入MIT格式的字典，返回结构化数据，并验证正则表达式"""
        """Load the MIT format dictionary, return structured data, and validate the regular expression."""
        with open(dic_path, encoding="utf8") as f:
            dic_lines = f.readlines()
            
        if len(dic_lines) == 0:
            return {}
            
        dic_path = os.path.abspath(dic_path)
        dic_name = os.path.basename(dic_path)
        dict_count = 0
        regex_errors = 0
        
        glossary_entries = {}
        
        for line_number, line in enumerate(dic_lines, start=1):
            line = line.strip()
            # 跳过空行和注释行 / Skip empty lines and comment lines
            if not line or line.startswith("#") or line.startswith("//"):
                continue
                
            # 处理注释 / Process comments
            comment = ""
            if '#' in line:
                parts = line.split('#', 1)
                line = parts[0].strip()
                comment = "#" + parts[1]
            elif '//' in line:
                parts = line.split('//', 1)
                line = parts[0].strip()
                comment = "//" + parts[1]
            
            # 先尝试用制表符分割源词和目标词
            # First, try to split the source word and target word using a tab character
            parts = line.split("\t", 1)
            if len(parts) == 1:  # 如果没有制表符，尝试用空格分割 / If there is no tab character, try to split using spaces
                parts = line.split(None, 1)  # None表示任何空白字符 / None represents any whitespace character
            
            if len(parts) < 2:
                # 只有一个单词，跳过或记录警告 / If there is only one word, skip it or log a warning
                self.logger.debug(f"Skipping lines with a single word: {line}")
                continue
            else:
                # 源词和目标词 / Source word and target word
                src = parts[0].strip()
                dst = parts[1].strip()
            
            # 验证正则表达式 / Validate the regular expression
            try:
                re.compile(src)
                # 正则表达式有效，将术语添加到字典中 / The regular expression is valid; add the term to the dictionary
                if comment:
                    entry = f"{dst} {comment}"
                else:
                    entry = dst
                
                glossary_entries[src] = entry
                dict_count += 1
            except re.error as e:
                # 正则表达式无效，记录错误 / The regular expression is invalid; log the error        
                regex_errors += 1
                error_message = str(e)
                self.logger.warning(f"Regular expression error on line {line_number}: '{src}' - {error_message}")
                
                # 提供修复建议 / Provide suggestions for fixes
                suggested_fix = src
                # 转义所有特殊字符 / Escape all special characters
                special_chars = {
                    '[': '\\[', ']': '\\]',
                    '(': '\\(', ')': '\\)',
                    '{': '\\{', '}': '\\}',
                    '.': '\\.', '*': '\\*',
                    '+': '\\+', '?': '\\?',
                    '|': '\\|', '^': '\\^',
                    '$': '\\$', '\\': '\\\\',
                    '/': '\\/'
                }
                
                for char, escaped in special_chars.items():
                    # 已经被转义的不处理 / Do not process characters that are already escaped
                    suggested_fix = re.sub(f'(?<!\\\\){re.escape(char)}', escaped, suggested_fix)
                
                # 特殊处理特定错误型 / Special handling for specific error types
                if "unterminated character set" in error_message:
                    # 如果是未闭合的字符集，查找最后一个'['并添加对应的']'
                    # If it is an unclosed character set, find the last '[' and add the corresponding ']'
                    last_open = suggested_fix.rfind('\\[')
                    if last_open != -1 and '\\]' not in suggested_fix[last_open:]:
                        suggested_fix += '\\]'
                
                elif "unbalanced parenthesis" in error_message:
                    # 如果是括号不平衡，检查并添加缺失的')'
                    # If the parentheses are unbalanced, check and add the missing ')'
                    open_count = suggested_fix.count('\\(')
                    close_count = suggested_fix.count('\\)')
                    if open_count > close_count:
                        suggested_fix += '\\)' * (open_count - close_count)
                    
                self.logger.info(f"Possible fix suggestions: '{suggested_fix}'")
        
        self.logger.info(f"Loading MIT format dictionary: {dic_name} containing {dict_count} entries, found {regex_errors} regular expression errors")
        return glossary_entries

    def load_galtransl_dic(self, dic_path):  
        """载入Galtransl格式的字典 / Loading a Galtransl format dictionary"""  
        glossary_entries = {}  
        
        try:  
            with open(dic_path, encoding="utf8") as f:  
                dic_lines = f.readlines()  
            
            if len(dic_lines) == 0:  
                return {}  
                
            dic_path = os.path.abspath(dic_path)  
            dic_name = os.path.basename(dic_path)  
            normalDic_count = 0  
            
            for line in dic_lines:  
                if line.startswith("\\\\") or line.startswith("//") or line.strip() == "":  
                    continue  
                
                # 尝试用制表符分割 / Attempting to split using tabs
                parts = line.split("\t")  
                # 如果分割结果不符合预期，尝试用空格分割 / If the split result is not as expected, try splitting using spaces    
                    
                if len(parts) != 2:  
                    parts = line.split("    ", 1)  # 四个空格 / Four spaces  
                
                if len(parts) == 2:  
                    src, dst = parts[0].strip(), parts[1].strip()  
                    glossary_entries[src] = dst  
                    normalDic_count += 1  
                else:  
                    self.logger.debug(f"Skipping lines that do not conform to the format.: {line.strip()}")  
            
            self.logger.info(f"Loading Galtransl format dictionary: {dic_name} containing {normalDic_count} entries")  
            return glossary_entries  
            
        except Exception as e:  
            self.logger.error(f"Error loading Galtransl dictionary: {e}")  
            return {}  

    def load_sakura_dict(self, dic_path):  
        """载入Sakura格式的字典 / Loading a Sakura format dictionary"""
        glossary_entries = {}  
        
        try:  
            with open(dic_path, encoding="utf8") as f:  
                dic_lines = f.readlines()  
            
            if len(dic_lines) == 0:  
                return {}  
                
            dic_path = os.path.abspath(dic_path)  
            dic_name = os.path.basename(dic_path)  
            dict_count = 0  
            
            for line in dic_lines:  
                line = line.strip()  
                if line.startswith("\\\\") or line.startswith("//") or line == "":  
                    continue  
                
                # Sakura格式使用 -> 分隔源词和目标词 /  
                # Sakura format uses -> to separate source words and target words
                if "->" in line:  
                    parts = line.split("->", 1)  
                    if len(parts) == 2:  
                        src, dst = parts[0].strip(), parts[1].strip()  
                        glossary_entries[src] = dst  
                        dict_count += 1  
                    else:  
                        self.logger.debug(f"Skipping lines that do not conform to the format: {line}")  
                else:  
                    self.logger.debug(f"Skipping lines that do not conform to the format: {line}")  
            
            self.logger.info(f"Loading Sakura format dictionary: {dic_name} containing {dict_count} entries")  
            return glossary_entries  
            
        except Exception as e:  
            self.logger.error(f"Error loading Sakura dictionary: {e}")  
            return {}       
            
    def extract_relevant_terms(self, text):  
        """自动提取和query相关的术语表条目，而不是一次性将术语表载入全部，以防止token浪费和系统提示词权重下降导致的指导效果减弱"""
        """Automatically extract glossary entries related to the query, 
           rather than loading the entire glossary at once, 
           to prevent token wastage and reduced guidance effectiveness due to a decrease in system prompt weight."""
        relevant_terms = {}  
        
        # 1. 编辑距离计算函数 / Edit distance calculation function 
        def levenshtein_distance(s1, s2):  
            if len(s1) < len(s2):  
                return levenshtein_distance(s2, s1)  
            if len(s2) == 0:  
                return len(s1)  
            
            previous_row = range(len(s2) + 1)  
            for i, c1 in enumerate(s1):  
                current_row = [i + 1]  
                for j, c2 in enumerate(s2):  
                    insertions = previous_row[j + 1] + 1  
                    deletions = current_row[j] + 1  
                    substitutions = previous_row[j] + (c1 != c2)  
                    current_row.append(min(insertions, deletions, substitutions))  
                previous_row = current_row  
            
            return previous_row[-1]  
        
        # 日语专用编辑距离计算 / Edit distance calculation specifically for Japanese  
        def japanese_levenshtein_distance(s1, s2):  
            # 先将两个字符串规范化为同一种写法 / First, normalize both strings to the same writing system. 
            s1 = normalize_japanese(s1)  
            s2 = normalize_japanese(s2)  
            # 计算规范化后的编辑距离 / Calculate the edit distance after normalization 
            return levenshtein_distance(s1, s2)  
        
        # 2. 日语文本规范化（将片假名转为平假名） / Japanese text normalization (convert katakana to hiragana)  
        def normalize_japanese(text):  
            result = ""  
            for char in text:  
                # 检查是否是片假名范围 (0x30A0-0x30FF)  
                # Check if it's within the katakana range (0x30A0-0x30FF)
                if 0x30A0 <= ord(char) <= 0x30FF:  
                    # 转换片假名到平假名 (减去0x60)  
                    # Convert katakana to hiragana (subtract 0x60)
                    hiragana_char = chr(ord(char) - 0x60)  
                    result += hiragana_char  
                else:  
                    result += char  
            return result  
        
        # 3. 增强的词规范化处理 / Enhanced word normalization processing          
        def normalize_term(term):
            # 基础处理 (Basic processing)
            term = re.sub(r'[^\w\s]', '', term)  # 移除标点符号 (Remove punctuation)
            term = term.lower()                   # 转换为小写 (Convert to lowercase)
            # 日语处理 (Japanese processing)
            term = normalize_japanese(term)       # 片假名转平假名 (Convert katakana to hiragana)
            return term
        
        # 4. 部分匹配函数 / Partial match function
        def partial_match(text, term):  
            normalized_text = normalize_term(text)  
            normalized_term = normalize_term(term)  
            return normalized_term in normalized_text  

        # 5. 日语特化的相似度判断 (Japanese-specific similarity judgment)
        def is_japanese_similar(text, term, threshold=2):
            # 规范化后计算编辑距离 (Calculate edit distance after normalization)
            normalized_text = normalize_term(text)
            normalized_term = normalize_term(term)

            # 如果术语很短，降低阈值 (Reduce the threshold if the term is short)
            if len(normalized_term) <= 4:
                threshold = 1

            # # 滑动窗口匹配（针对较长文本和短术语）- 可能过拟合，需要进一步调整 (Sliding window matching (for longer texts and short terms) - May overfit, needs further adjustment)
            # if len(normalized_text) > len(normalized_term):
            #     min_distance = float('inf')
            #     # 创建与术语等长的窗口，在文本中滑动 (Create a window of the same length as the term and slide it through the text)
            #     for i in range(len(normalized_text) - len(normalized_term) + 1):
            #         window = normalized_text[i:i+len(normalized_term)]
            #         distance = japanese_levenshtein_distance(window, normalized_term)
            #         min_distance = min(min_distance, distance)
            #     return min_distance <= threshold
            # else:
            #     # 直接计算编辑距离 (Calculate the edit distance directly)
            #     distance = japanese_levenshtein_distance(normalized_text, normalized_term)
            #     return distance <= threshold

            # 直接计算编辑距离 (Calculate the edit distance directly)
            distance = japanese_levenshtein_distance(normalized_text, normalized_term)
            return distance <= threshold
      
        # 主匹配逻辑 (Main matching logic)
        for term, translation in self.glossary_entries.items():
            # 1. 精确匹配 (Exact match)
            if term in text:
                relevant_terms[term] = translation
                continue

            # 2. 日语特化的相似度匹配 (Japanese-specific similarity matching)
            if any(c for c in term if 0x3040 <= ord(c) <= 0x30FF):  # 检查是否包含日语字符 (Check if it contains Japanese characters)
                if is_japanese_similar(text, term):
                    relevant_terms[term] = translation
                    continue

            # 3. 普通编辑距离匹配（非日语文本） (Ordinary edit distance matching (non-Japanese text))
            normalized_text = normalize_term(text)
            normalized_term = normalize_term(term)

            # 4. 部分匹配 (Partial match)
            if partial_match(text, term):
                relevant_terms[term] = translation
                continue

            # 5. 正则表达式匹配 (Regular expression matching)
            pattern = re.compile(term, re.IGNORECASE)
            if pattern.search(text):
                relevant_terms[term] = translation

        return relevant_terms 
