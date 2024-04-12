from distutils.cygwinccompiler import get_versions
import re
import os
from venv import logger

from httpx import get

try:
    import openai
    import openai.error
except ImportError:
    openai = None
import asyncio
from typing import List, Dict

from .common import CommonTranslator
from .keys import SAKURA_API_BASE, SAKURA_VERSION, SAKURA_DICT_PATH

import logging

class SakuraDict():
    def __init__(self, path: str, logger: logging.Logger):
        self.logger = logger
        self.path = path
        self.dict_str = ""
        self.dict_str = self.get_dict_from_file(path)

    def load_galtransl_dic(self, dic_path: str):
        """
        载入Galtransl词典。
        """

        with open(dic_path, encoding="utf8") as f:
            dic_lines = f.readlines()
        if len(dic_lines) == 0:
            return
        dic_path = os.path.abspath(dic_path)
        dic_name = os.path.basename(dic_path)
        normalDic_count = 0

        gpt_dict = []
        for line in dic_lines:
            if line.startswith("\n"):
                continue
            elif line.startswith("\\\\") or line.startswith("//"):  # 注释行跳过
                continue

            # 四个空格换成Tab
            line = line.replace("    ", "\t")

            sp = line.rstrip("\r\n").split("\t")  # 去多余换行符，Tab分割
            len_sp = len(sp)

            if len_sp < 2:  # 至少是2个元素
                continue

            src = sp[0]
            dst = sp[1]
            info = sp[2] if len_sp > 2 else None
            gpt_dict.append({"src": src, "dst": dst, "info": info})
            normalDic_count += 1

        gpt_dict_text_list = []
        for gpt in gpt_dict:
            src = gpt['src']
            dst = gpt['dst']
            info = gpt['info'] if "info" in gpt.keys() else None
            if info:
                single = f"{src}->{dst} #{info}"
            else:
                single = f"{src}->{dst}"
            gpt_dict_text_list.append(single)

        gpt_dict_raw_text = "\n".join(gpt_dict_text_list)
        self.dict_str = gpt_dict_raw_text
        self.logger.info(
            f"载入 Galtransl 字典: {dic_name} {normalDic_count}普通词条"
        )

    def load_sakura_dict(self, dic_path: str):
        """
        直接载入标准的Sakura字典。
        """

        with open(dic_path, encoding="utf8") as f:
            dic_lines = f.readlines()

        self.logger.debug(f"载入Sakura字典: {dic_path}")
        self.logger.debug(f"载入Sakura字典: {dic_lines}")
        if len(dic_lines) == 0:
            return
        dic_path = os.path.abspath(dic_path)
        dic_name = os.path.basename(dic_path)
        normalDic_count = 0

        gpt_dict_text_list = []
        for line in dic_lines:
            if line.startswith("\n"):
                continue
            elif line.startswith("\\\\") or line.startswith("//"):  # 注释行跳过
                continue

            sp = line.rstrip("\r\n").split("->")  # 去多余换行符，->分割
            len_sp = len(sp)

            if len_sp < 2:  # 至少是2个元素
                continue

            src = sp[0]
            dst_info = sp[1].split("#")  # 使用#分割目标和信息
            dst = dst_info[0].strip()
            info = dst_info[1].strip() if len(dst_info) > 1 else None
            if info:
                single = f"{src}->{dst} #{info}"
            else:
                single = f"{src}->{dst}"
            gpt_dict_text_list.append(single)
            normalDic_count += 1

        gpt_dict_raw_text = "\n".join(gpt_dict_text_list)
        self.dict_str = gpt_dict_raw_text
        self.logger.debug(
            f"载入标准Sakura字典: {dic_name} {normalDic_count}普通词条"
        )

    def detect_type(self, dic_path: str):
        """
        检测字典类型。
        """
        with open(dic_path, encoding="utf8") as f:
            dic_lines = f.readlines()
        self.logger.debug(f"检测字典类型: {dic_path}")
        if len(dic_lines) == 0:
            return "unknown"

        # 判断是否为Galtransl字典
        is_galtransl = True
        for line in dic_lines:
            if line.startswith("\n"):
                continue
            elif line.startswith("\\\\") or line.startswith("//"):
                continue

            if "\t" not in line and "    " not in line:
                is_galtransl = False
                break

        if is_galtransl:
            return "galtransl"

        # 判断是否为Sakura字典
        is_sakura = True
        for line in dic_lines:
            if line.startswith("\n"):
                continue
            elif line.startswith("\\\\") or line.startswith("//"):
                continue

            if "->" not in line:
                is_sakura = False
                break

        if is_sakura:
            return "sakura"

        return "unknown"

    def get_dict_str(self):
        """
        获取字典内容。
        """
        if self.dict_str == "":
            self.logger.warning("字典为空")
            return ""
        return self.dict_str

    def get_dict_from_file(self, dic_path: str):
        """
        从文件载入字典。
        """
        dic_type = self.detect_type(dic_path)
        if dic_type == "galtransl":
            self.load_galtransl_dic(dic_path)
        elif dic_type == "sakura":
            self.load_sakura_dict(dic_path)
        else:
            self.logger.warning(f"未知的字典类型: {dic_path}")
        return self.get_dict_str()


class SakuraTranslator(CommonTranslator):

    _TIMEOUT = 999  # 等待服务器响应的超时时间(秒)
    _RETRY_ATTEMPTS = 1  # 请求出错时的重试次数
    _TIMEOUT_RETRY_ATTEMPTS = 3  # 请求超时时的重试次数
    _RATELIMIT_RETRY_ATTEMPTS = 3  # 请求被限速时的重试次数

    _CHAT_SYSTEM_TEMPLATE_009 = (
        '你是一个轻小说翻译模型，可以流畅通顺地以日本轻小说的风格将日文翻译成简体中文，并联系上下文正确使用人称代词，不擅自添加原文中没有的代词。'
    )
    _CHAT_SYSTEM_TEMPLATE_010 = (
        '你是一个轻小说翻译模型，可以流畅通顺地以日本轻小说的风格将日文翻译成简体中文，并联系上下文正确使用人称代词，注意不要擅自添加原文中没有的代词，也不要擅自增加或减少换行。'
    )

    _LANGUAGE_CODE_MAP = {
        'CHS': 'Simplified Chinese',
        'JPN': 'Japanese'
    }

    def __init__(self):
        super().__init__()
        if "/v1" not in SAKURA_API_BASE:
            openai.api_base = SAKURA_API_BASE + "/v1"
        else:
            openai.api_base = SAKURA_API_BASE
        openai.api_key = "sk-114514"
        self.temperature = 0.3
        self.top_p = 0.3
        self.frequency_penalty = 0.1
        self._current_style = "normal"
        self._emoji_pattern = re.compile(r'[\U00010000-\U0010ffff]')
        self._heart_pattern = re.compile(r'❤')
        self.sakura_dict = SakuraDict(self.get_dict_path(), self.logger)

    def get_sakura_version(self):
        return SAKURA_VERSION

    def get_dict_path(self):
        return SAKURA_DICT_PATH

    def detect_and_remove_extra_repeats(self, s: str, threshold: int = 20, remove_all=True):
        """
        检测字符串中是否有任何模式连续重复出现超过阈值，并在去除多余重复后返回新字符串。
        保留一个模式的重复。
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
                    if remove_all:
                        s = s[:i + pattern_length] + s[j:]
                    break
                i += 1
            if repeated:
                break
        return repeated, s

    def _format_prompt_log(self, prompt: str) -> str:
        """
        格式化日志输出的提示文本。
        """
        gpt_dict_raw_text = self.sakura_dict.get_dict_str()
        prompt_009 = '\n'.join([
            'System:',
            self._CHAT_SYSTEM_TEMPLATE_009,
            'User:',
            '将下面的日文文本翻译成中文：',
            prompt,
        ])
        prompt_010 = '\n'.join([
            'System:',
            self._CHAT_SYSTEM_TEMPLATE_010,
            'User:',
            "根据以下术语表：",
            gpt_dict_raw_text,
            "将下面的日文文本根据上述术语表的对应关系和注释翻译成中文：",
            prompt,
        ])
        return prompt_009 if SAKURA_VERSION == '0.0.9' else prompt_010
    
    def _split_text(self, text: str) -> List[str]:
        """
        将字符串按换行符分割为列表。
        """
        if isinstance(text, list):
            return text
        return text.split('\n')

    def _preprocess_queries(self, queries: List[str]) -> List[str]:
        """
        预处理查询文本,去除emoji,替换特殊字符,并添加「」标记。
        """
        queries = [self._emoji_pattern.sub('', query) for query in queries]
        queries = [self._heart_pattern.sub('♥', query) for query in queries]
        queries = [f'「{query}」' for query in queries]
        return queries

    async def _check_translation_quality(self, queries: List[str], response: str) -> List[str]:
        """
        检查翻译结果的质量,包括重复和行数对齐问题,如果存在问题则尝试重新翻译或返回原始文本。
        """
        rep_flag = self._detect_repeats(response)
        if rep_flag:
            for i in range(self._RETRY_ATTEMPTS):
                if self._detect_repeats(''.join(queries)):
                    self.logger.warning('Queries have repeats.')
                    break
                self.logger.warning(f'Re-translating due to model degradation, attempt: {i + 1}')
                self._set_gpt_style("precise")
                response = await self._handle_translation_request(queries)
                rep_flag = self._detect_repeats(response)
                if not rep_flag:
                    break
            if rep_flag:
                self.logger.warning('Model degradation, translating single lines.')
                return await self._translate_single_lines(queries)

        align_flag = self._check_align(queries, response)
        if not align_flag:
            for i in range(self._RETRY_ATTEMPTS):
                self.logger.warning(f'Re-translating due to mismatched lines, attempt: {i + 1}')
                self._set_gpt_style("precise")
                response = await self._handle_translation_request(queries)
                align_flag = self._check_align(queries, response)
                if align_flag:
                    break
            if not align_flag:
                self.logger.warning('Mismatched lines, translating single lines.')
                return await self._translate_single_lines(queries)

        return self._split_text(response)

    def _detect_repeats(self, text: str, threshold: int = 20) -> bool:
        """
        检测文本中是否存在重复模式。
        """
        _, text = self.detect_and_remove_extra_repeats(text, threshold)
        return text != text

    def _check_align(self, queries: List[str], response: str) -> bool:
        """
        检查原始文本和翻译结果的行数是否对齐。
        """
        translations = self._split_text(response)
        is_aligned = len(queries) == len(translations)
        if not is_aligned:
            self.logger.warning(f"Mismatched lines - Queries: {len(queries)}, Translations: {len(translations)}")
        return is_aligned

    async def _translate_single_lines(self, queries: List[str]) -> List[str]:
        """
        逐行翻译查询文本。
        """
        translations = []
        for query in queries:
            response = await self._handle_translation_request(query)
            if self._detect_repeats(response):
                self.logger.warning('Model degradation, using original text.')
                translations.append(query)
            else:
                translations.append(response)
        return translations

    def _delete_quotation_mark(self, texts: List[str]) -> List[str]:
        """
        删除文本中的「」标记。
        """
        new_texts = []
        for text in texts:
            text = text.strip('「」')
            new_texts.append(text)
        return new_texts

    async def _translate(self, from_lang: str, to_lang: str, queries: List[str]) -> List[str]:
        self.logger.debug(f'Temperature: {self.temperature}, TopP: {self.top_p}')
        self.logger.debug(f'Queries: {queries}')
        text_prompt = '\n'.join(queries)
        self.logger.debug('-- Sakura Prompt --\n' + self._format_prompt_log(text_prompt) + '\n\n')

        # 预处理查询文本
        queries = self._preprocess_queries(queries)

        # 发送翻译请求
        response = await self._handle_translation_request(queries)
        self.logger.debug('-- Sakura Response --\n' + response + '\n\n')

        # 检查翻译结果是否存在重复或行数不匹配的问题
        translations = await self._check_translation_quality(queries, response)

        return self._delete_quotation_mark(translations)

    async def _handle_translation_request(self, prompt: str) -> str:
        """
        处理翻译请求,包括错误处理和重试逻辑。
        """
        ratelimit_attempt = 0
        server_error_attempt = 0
        timeout_attempt = 0
        while True:
            try:
                request_task = asyncio.create_task(self._request_translation(prompt))
                response = await asyncio.wait_for(request_task, timeout=self._TIMEOUT)
                break
            except asyncio.TimeoutError:
                timeout_attempt += 1
                if timeout_attempt >= self._TIMEOUT_RETRY_ATTEMPTS:
                    raise Exception('Sakura timeout.')
                self.logger.warning(f'Restarting request due to timeout. Attempt: {timeout_attempt}')
            except openai.error.RateLimitError:
                ratelimit_attempt += 1
                if ratelimit_attempt >= self._RATELIMIT_RETRY_ATTEMPTS:
                    raise
                self.logger.warning(f'Restarting request due to ratelimiting by sakura servers. Attempt: {ratelimit_attempt}')
                await asyncio.sleep(2)
            except (openai.error.APIError, openai.error.APIConnectionError) as e:
                server_error_attempt += 1
                if server_error_attempt >= self._RETRY_ATTEMPTS:
                    self.logger.error(f'Sakura server error: {str(e)}. Returning original text.')
                    return prompt
                self.logger.warning(f'Restarting request due to server error. Attempt: {server_error_attempt}')

        return response

    async def _request_translation(self, input_text_list) -> str:
        """
        向Sakura API发送翻译请求。
        """
        if isinstance(input_text_list, list):
            raw_text = "\n".join(input_text_list)
        else:
            raw_text = input_text_list
        extra_query = {
            'do_sample': False,
            'num_beams': 1,
            'repetition_penalty': 1.0,
        }
        if SAKURA_VERSION == "0.9":
            messages=[
                    {
                        "role": "system",
                        "content": f"{self._CHAT_SYSTEM_TEMPLATE_009}"
                    },
                    {
                        "role": "user",
                        "content": f"将下面的日文文本翻译成中文：{raw_text}"
                    }
                ]
        else:
            gpt_dict_raw_text = self.sakura_dict.get_dict_str()
            self.logger.debug(f"Sakura Dict: {gpt_dict_raw_text}")
            messages=[
                    {
                        "role": "system",
                        "content": f"{self._CHAT_SYSTEM_TEMPLATE_010}"
                    },
                    {
                        "role": "user",
                        "content": f"根据以下术语表：\n{gpt_dict_raw_text}\n将下面的日文文本根据上述术语表的对应关系和注释翻译成中文：{raw_text}"
                    }
                ]
        response = await openai.ChatCompletion.acreate(
            model="sukinishiro",
            messages=messages,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=1024,
            frequency_penalty=self.frequency_penalty,
            seed=-1,
            extra_query=extra_query,
        )
        # 提取并返回响应文本
        for choice in response.choices:
            if 'text' in choice:
                return choice.text

        return response.choices[0].message.content

    def _set_gpt_style(self, style_name: str):
        """
        设置GPT的生成风格。
        """
        if self._current_style == style_name:
            return
        self._current_style = style_name
        if style_name == "precise":
            temperature, top_p = 0.1, 0.3
            frequency_penalty = 0.1
        elif style_name == "normal":
            temperature, top_p = 0.3, 0.3
            frequency_penalty = 0.15

        self.temperature = temperature
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
