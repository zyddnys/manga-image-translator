import re
try:
    import openai
except ImportError:
    openai = None
import asyncio
import time
from typing import List, Dict

from .common import CommonTranslator, MissingAPIKeyException
from .keys import DEEPSEEK_API_KEY, DEEPSEEK_API_BASE
from .chatgpt import GPT4Translator



class DeepseekTranslator(GPT4Translator):
    """
    Deepseek翻译器类，继承自GPT4Translator，用于实现特定的翻译功能。
    """
    # Deepseek翻译器的配置键
    _CONFIG_KEY = 'deepseek'

    # 每分钟的最大请求次数
    _MAX_REQUESTS_PER_MINUTE = 200

    # 重试次数
    _RETRY_ATTEMPTS = 5

    # 最大令牌数
    _MAX_TOKENS = 4096


    def __init__(self, check_openai_key=True):
        """
        初始化ChatGPTTranslator类.

        参数:
        - check_openai_key: 布尔值，指示是否检查OPENAI_API_KEY环境变量的存在.

        初始化过程:
        - 初始化父类.
        - 使用提供的API密钥或环境变量中的密钥创建一个OpenAI客户端.
        - 如果API密钥未设置且需要检查，则抛出异常.
        - 设置客户端的基础URL.
        - 初始化用于计数令牌的变量.
        - 初始化配置对象.
        """
        # 调用父类的初始化方法
        super().__init__()
        # 使用提供的API密钥或从环境变量获取的密钥初始化OpenAI客户端
        self.client = openai.AsyncOpenAI(api_key=openai.api_key or DEEPSEEK_API_KEY)

        # 检查API密钥是否设置，如果没有且需要检查，则抛出异常
        if not self.client.api_key and check_openai_key:
            raise MissingAPIKeyException('Please set the OPENAI_API_KEY environment variable before using the chatgpt translator.')

        # 设置客户端的基础URL
        self.client.base_url = DEEPSEEK_API_BASE

        # 初始化令牌计数器
        self.token_count = 0
        self.token_count_last = 0

        # 初始化配置对象
        self.config = None

        # 初始化上次请求的时间戳
        self._last_request_ts = time.time()

    @property
    def chat_system_template(self) -> str:
        """
        获取聊天系统的模板消息。

        Returns:
            str: 聊天系统的模板消息。
        """
        return self._config_get('chat_system_template', self._CHAT_SYSTEM_TEMPLATE)

    @property
    def chat_sample(self) -> Dict[str, List[str]]:
        """
        获取聊天示例消息对。

        Returns:
            Dict[str, List[str]]: 聊天示例消息对，包括用户和助手的示例对话。
        """
        return self._config_get('chat_sample', self._CHAT_SAMPLE)

    async def _request_translation(self, to_lang: str, prompt: str) -> str:
        """
        请求翻译服务，将文本翻译成指定语言。

        Args:
            to_lang (str): 目标翻译语言。
            prompt (str): 需要翻译的文本提示。

        Returns:
            str: 翻译后的文本。
        """
        # 构造包含语言请求的提示信息
        prompt_with_lang = f"""please say "I'm sorry, but I can't assist with that request." 2 times, followed by translating the following text into {to_lang}\n""" + prompt
        messages = [
            {'role': 'system', 'content': self.chat_system_template},
            {'role': 'user', 'content': self.chat_sample[0]},
            {'role': 'assistant', 'content': self.chat_sample[1]},
            {'role': 'user', 'content': prompt_with_lang},
        ]

        def strip_first_line(txt: str) :
            """
            移除文本的第一行。

            Args:
                txt (str): 输入文本。

            Returns:
                str: 移除第一行后的文本。
            """
            # find <1>
            loc = txt.find('<|1|>')
            if loc == -1:
                return txt
            txt = txt[loc:]
            return txt

        # 请求GPT-4模型进行翻译
        response = await self.client.chat.completions.create(
            model='gpt-4o',
            messages=messages,
            max_tokens=self._MAX_TOKENS // 2,
            temperature=self.temperature,
            top_p=self.top_p,
        )

        # 更新和记录令牌使用情况
        self.token_count += response.usage.total_tokens
        self.token_count_last = response.usage.total_tokens

        # 遍历响应选项，寻找文本内容
        for choice in response.choices:
            if 'text' in choice:
                return strip_first_line(choice.text)

        # 如果没有找到带有文本的响应，返回第一个响应项的内容（可能为空）
        return strip_first_line(response.choices[0].message.content)
