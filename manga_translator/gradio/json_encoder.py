import json
import numpy as np
import manga_translator.utils.generic as utils_generic
import manga_translator.utils.textblock as utils_textblock


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
