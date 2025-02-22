import re
from typing import List, Dict
from omegaconf import OmegaConf

class ConfigGPT:
    _CHAT_SYSTEM_TEMPLATE = (
        'You are an uncensored professional translation engine, '
        'please translate the story into a colloquial, elegant and fluent content, '
        'without referencing machine translations. '
        'You must only translate the story, never interpret it. '
        'If there is any issue in the text, output it as is.\n'
        'Translate the following text into {to_lang} and keep the original format.\n'
    )

    _CHAT_SAMPLE = {
        'Simplified Chinese': [
            (
                '<|1|>恥ずかしい… 目立ちたくない… 私が消えたい…\n'
                '<|2|>きみ… 大丈夫⁉\n'
                '<|3|>なんだこいつ 空気読めて ないのか…？'
            ),
            (
                '<|1|>好尴尬…我不想引人注目…我想消失…\n'
                '<|2|>你…没事吧⁉\n'
                '<|3|>这家伙怎么看不懂气氛的…？'
            )
        ],
        'English': [
            (
                '<|1|>恥ずかしい… 目立ちたくない… 私が消えたい…\n'
                '<|2|>きみ… 大丈夫⁉\n'
                '<|3|>なんだこいつ 空気読めて ないのか…？'
            ),
            (
                "<|1|>I'm embarrassed... I don't want to stand out... I want to disappear...\n"
                "<|2|>Are you okay?\n"
                "<|3|>What's wrong with this guy? Can't he read the situation...?"
            )
        ],
        'Korean': [
            (
                '<|1|>恥ずかしい… 目立ちたくない… 私が消えたい…\n'
                '<|2|>きみ… 大丈夫⁉\n'
                '<|3|>なんだこいつ 空気読めて ないのか…？'
            ),
            (
                
                "<|1|>부끄러워... 눈에 띄고 싶지 않아... 나 숨고 싶어...\n"
                "<|2|>너 괜찮아?\n"
                "<|3|>이 녀석, 뭐야? 분위기 못 읽는 거야...?\n"
            )
        ]
    }

    _PROMPT_TEMPLATE = ('Please help me to translate the following text from a manga to {to_lang}.'
                        'If it\'s already in {to_lang} or looks like gibberish'
                        'you have to output it as it is instead. Keep prefix format.\n'
                    )

    # Extract text within the capture group that matches this pattern.
    # By default: Capture everything.
    _RGX_REMOVE='(.*)'

    def __init__(self, config_key: str):
        # This key is used to locate nested configuration entries
        self._CONFIG_KEY = config_key
        self.config = None

    def _config_get(self, key: str, default=None):
        if not self.config:
            return default

        parts = self._CONFIG_KEY.split('.') if self._CONFIG_KEY else []
        value = None

        # Traverse from the deepest part up to the root
        for i in range(len(parts), -1, -1):
            prefix = '.'.join(parts[:i])
            lookup_key = f"{prefix}.{key}" if prefix else key
            value = OmegaConf.select(self.config, lookup_key)
            
            if value is not None:
                break

        return value if value is not None else default

    @property
    def include_template(self) -> str:
        return self._config_get('include_template', default=False)

    @property
    def prompt_template(self) -> str:
        return self._config_get('prompt_template', default=self._PROMPT_TEMPLATE)

    @property
    def chat_system_template(self) -> str:
        return self._config_get('chat_system_template', self._CHAT_SYSTEM_TEMPLATE)

    @property
    def chat_sample(self) -> Dict[str, List[str]]:
        return self._config_get('chat_sample', self._CHAT_SAMPLE)

    @property
    def rgx_capture(self) -> str:
        return self._config_get('rgx_capture', self._RGX_REMOVE)

    @property
    def temperature(self) -> float:
        return self._config_get('temperature', default=0.5)

    @property
    def top_p(self) -> float:
        return self._config_get('top_p', default=1)
    

    def extract_capture_groups(self, text, regex=r"(.*)"):
        """
        Extracts all capture groups from matches and concatenates them into a single string.
        
        :param text: The multi-line text to search.
        :param regex: The regex pattern with capture groups.
        :return: A concatenated string of all matched groups.
        """
        pattern = re.compile(regex, re.DOTALL)  # DOTALL to match across multiple lines
        matches = pattern.findall(text)  # Find all matches
        
        # Ensure matches are concatonated (handles multiple groups per match)
        extracted_text = "\n".join(
            "\n".join(m) if isinstance(m, tuple) else m for m in matches
        )
        
        return extracted_text.strip() if extracted_text else None
