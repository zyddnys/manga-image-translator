# consume tasks from moeflow job worker with manga-image-translator code
import re
from typing import Any, Awaitable

from celery import Celery
from asgiref.sync import async_to_sync
import manga_translator.detection as detection
import manga_translator.ocr as ocr
import manga_translator.textline_merge as textline_merge
import manga_translator.utils.generic as utils_generic
import manga_translator.utils.textblock as utils_textblock

import logging
import json
import os
import dotenv

from PIL import Image
import numpy as np

dotenv.load_dotenv()
BROKER_URL = os.environ.get('CELERY_BROKER_URL')
BACKEND_URL = os.environ.get('CELERY_BACKEND_URL')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

env = {
    "RABBITMQ_USER": 'moeflow',
    "RABBITMQ_PASS": 'PLEASE_CHANGE_THIS',
    'RABBITMQ_VHOST_NAME': 'moeflow',
    'MONGODB_USER': 'moeflow',
    'MONGODB_PASS': 'PLEASE_CHANGE_THIS',
    'MONGODB_DB_NAME': 'moeflow',
}

celery_app = Celery(
    "manga-image-translator-moeflow-worker",
    broker=BROKER_URL,
    backend=BACKEND_URL,
    result_expires = 7 * 24 * 60 * 60, # 7d
)


class JSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, utils_textblock.TextBlock):
            return {
                'pts': o.lines,
                'text': o.text,
                'textlines': self.default(o.texts),
            }
        if isinstance(o, utils_generic.Quadrilateral):
            return {
                'pts': o.pts,
                'text': o.text,
                'prob': o.prob,
                'textlines': self.default(o.textlines),
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


def to_json(value: object) -> Any:
    """
    :return: a json-serizable deep clone of `value`
    """
    return json.loads(json.dumps(value, cls=JSONEncoder))


@celery_app.task(name="tasks.mit.detect_text")
def mit_detect_text(path_or_url: str, **kwargs):
    logger.debug("Running text segmentation %s %s", path_or_url, kwargs)
    result = async_detection(path_or_url, **kwargs)
    logger.debug("Running text segmentation result = %s", result)
    return result


# OCR + detect_textblocks + merge_textlines
@celery_app.task(name='tasks.mit.ocr')
def mit_ocr(path_or_url: str, **kwargs):
    logger.debug("Running OCR %s %s", path_or_url, kwargs)
    # for unknown reason async_ocr returns [[Quad]] instead of [result]
    textlines = async_ocr(path_or_url, **kwargs)
    # return json.loads(json.dumps(result, cls=JSONEncoder)),
    logger.debug("Running OCR result = %s", textlines)

    img_w, img_h, *_rest = load_rgb_image(path_or_url).shape

    min_text_length = kwargs.get('min_text_length', 0)
    text_blocks_all: list[utils_textblock.TextBlock] = async_textline_merge(
        textlines=textlines,
        width=img_w,
        height=img_h)

    # logger.debug("text_blocks_all = %s", text_regions_all)
    text_blocks = filter(
        lambda r: len(r.text) > min_text_length and utils_generic.is_valuable_text(r.text),
        text_blocks_all)

    return to_json(text_blocks)


@celery_app.task(name='tasks.mit.translate')
def mit_translate(**kwargs):
    logger.debug("Running translate %s", kwargs)
    result = async_translate(**kwargs)
    logger.debug("Running translate result = %s", result)
    return result


@celery_app.task(name='tasks.mit.inpaint')
def mit_inpaint(path_or_url: str, **kwargs):
    raise NotImplementedError()


def load_rgb_image(path_or_url: str) -> np.ndarray:
    if re.match(r'^https?://', path_or_url):
        raise NotImplementedError("URL not supported yet")
    img = Image.open(path_or_url)
    img_rgb, img_alpha = utils_generic.load_image(img)
    return img_rgb


def deserialize_quad_list(text_lines: list[dict]) -> list[utils_generic.Quadrilateral]:
    def create(json_value: dict) -> utils_generic.Quadrilateral:
        optional_args = {
            k: json_value[k]
            for k in ['fg_r', 'fg_g', 'fg_b', 'bg_r', 'bg_g', 'bg_b']
            if k in json_value
        }
        return utils_generic.Quadrilateral(
            pts=np.array(json_value['pts']),
            text=json_value['text'],
            prob=json_value['prob'],
            **optional_args
        )

    return list(map(create, text_lines))


@async_to_sync
async def async_detection(path_or_url: str, **kwargs: str):
    await detection.prepare(kwargs['detector_key'])
    img = load_rgb_image(path_or_url)
    textlines, mask_raw, mask = await detection.dispatch(
        image=img,
        # detector_key=kwargs['detector_key'],
        **kwargs,
    )
    return {
        'textlines': json.loads(json.dumps(textlines, cls=JSONEncoder)),
        # 'mask_raw': mask_raw,
        # 'mask': mask,
    }


@async_to_sync
async def async_ocr(path_or_url: str, **kwargs) -> Awaitable[list[utils_generic.Quadrilateral]]:
    await ocr.prepare(kwargs['ocr_key'])
    img = load_rgb_image(path_or_url)
    quads = deserialize_quad_list(kwargs['regions'])
    result: list[utils_generic.Quadrilateral] = await ocr.dispatch(
        ocr_key=kwargs['ocr_key'],
        image=img,
        regions=quads,
        args=kwargs,
        verbose=kwargs.get('verbose', False),
    )
    return result


@async_to_sync
async def async_textline_merge(*, textlines: list[utils_generic.Quadrilateral], width: int, height: int) \
        -> list[utils_textblock.TextBlock]:
    return await textline_merge.dispatch(textlines, width, height)


@async_to_sync
async def async_translate(**kwargs) -> Awaitable[list[str]]:
    # FIXME: impl better translator , maybe with Langchain
    import manga_translator.translators as translators
    query = kwargs['query']
    target_lang = kwargs['target_lang']
    translator = translators.get_translator(kwargs['translator'])
    if isinstance(translator, translators.OfflineTranslator):
        await translator.download()
        await translator.load('auto', target_lang, device='cpu')
    result = await translator.translate(
        from_lang='auto',
        to_lang=target_lang,
        queries=[query],
    )
    return result
