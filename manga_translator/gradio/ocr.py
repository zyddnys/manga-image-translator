import manga_translator.textline_merge as textline_merge
import manga_translator.utils.textblock as utils_textblock
from dataclasses import dataclass
from typing import List, Optional
from .json_encoder import to_json

mit_ocr_default_params = dict(
    ocr_key="48px",  # recommended by rowland
    # ocr_key="48px_ctc",
    # ocr_key="mocr",  # XXX: mocr may have different output format
    # use_mocr_merge=True,
    verbose=True,
)


@dataclass(frozen=True, kw_only=True)
class OcrState:
    text_blocks: Optional[List[utils_textblock.TextBlock]] = None
    ocr_key: Optional[str] = None

    def copy(self, **kwargs):
        return OcrState(
            ocr_key=kwargs.get("ocr_key", self.ocr_key),
            text_blocks=kwargs.get("text_blocks", self.text_blocks),
        )
