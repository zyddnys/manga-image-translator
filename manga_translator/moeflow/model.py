import json
from pathlib import Path
import numpy as np
import manga_translator.utils.generic as utils_generic
import manga_translator.utils.textblock as utils_textblock
from pydantic import BaseModel


class TextBlock(BaseModel):
    xyxy: tuple[int, int, int, int]
    text: str
    textlines: list[str]

    @classmethod
    def from_mit(cls, textblock: utils_textblock.TextBlock):
        return cls(xyxy=textblock.xyxy, text=textblock.text, textlines=textblock.texts)


class FileProcessResult(BaseModel):
    local_path: Path
    text_blocks: list[TextBlock]
    translated: dict[str, list[str]] | None = None
    ocr_key: str | None = None
    detector_key: str | None = None


class FileBatchProcessResult(BaseModel):
    project_name: str
    files: list[FileProcessResult]
    target_languages: list[str] | None = None


class JSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, utils_textblock.TextBlock):
            return {
                "pts": o.lines,
                "text": o.text,
                "textlines": self.default(o.texts),
            }
        if isinstance(o, utils_generic.Quadrilateral):
            return {
                "pts": o.pts,
                "text": o.text,
                "prob": o.prob,
                "textlines": self.default(o.textlines),
            }
        elif isinstance(o, filter) or isinstance(o, tuple):
            return self.default(list(o))
        elif isinstance(o, list):
            return o
        elif isinstance(o, str):
            return o
        elif isinstance(o, np.ndarray):
            return o.tolist()
        elif isinstance(o, np.integer):
            return int(o)
        elif isinstance(o, np.floating):
            return float(o)
        else:
            return super().default(o)


def to_json(obj) -> object:
    return json.loads(json.dumps(obj, cls=JSONEncoder))
