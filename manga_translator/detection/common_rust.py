import numpy as np
from rusty_manga_image_translator import Session, PyPreprocessorOptions, PyDefaultOptions, PyImage

from manga_translator.config import TranslatorConfig
from manga_translator.utils import Quadrilateral

def get_session():
    """Gets onnx session"""
    if not hasattr(get_session, "_instance"):
        get_session._instance = Session(None)
    return get_session._instance


class RustDetector:
    def __init__(self, det):
        self.det = det

    def parse_args(self, args: TranslatorConfig):
        # parse_args does noting
        pass

    def is_downloaded(self) -> bool:
        #do not use python downloader
        return True

    async def download(self, force=False):
        # download does nothing
        pass

    @property
    def model_dir(self):
        # should not be used
        raise NotImplementedError

    async def infer(self, *args, **kwargs):
        # should not be used
        raise NotImplementedError

    async def detect(self, image: np.ndarray, detect_size: int, text_threshold: float, box_threshold: float,
                     unclip_ratio: float,
                     invert: bool, gamma_correct: bool, rotate: bool, auto_rotate: bool = False, verbose: bool = False):
        '''
        Returns textblock list and text mask.
        '''
        o1 = PyPreprocessorOptions(invert, gamma_correct, rotate, auto_rotate)
        o2 = PyDefaultOptions(detect_size, unclip_ratio, text_threshold, box_threshold)
        img = PyImage.from_numpy(image)
        areas, raw_mask = self.det.detect(img, o1, o2)
        textlines = [Quadrilateral(np.array(x.pts(), dtype=np.int64), '', x.score()) for x in areas]
        return textlines, raw_mask, None

    async def reload(self, device: str, *args, **kwargs):
        await self.unload()
        await self.load(device)

    async def load(self, device: str, *args, **kwargs):
        self.det.load()

    async def unload(self):
        self.det.unload()

    def is_loaded(self) -> bool:
        return self.det.loaded()