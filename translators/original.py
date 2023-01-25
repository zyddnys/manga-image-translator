from typing import List

from translators.common import CommonTranslator

class OriginalTranslator(CommonTranslator):
    
    def supports_languages(self, from_lang: str, to_lang: str, fatal: bool = False) -> bool:
        return True

    async def _translate(self, from_lang: str, to_lang: str, queries: List[str]) -> List[str]:
        return queries
