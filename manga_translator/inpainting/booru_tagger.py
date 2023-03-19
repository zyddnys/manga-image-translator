import os
import gc
import pandas as pd
import numpy as np
from onnxruntime import InferenceSession
from typing import Tuple, List, Dict
from io import BytesIO
from PIL import Image

import cv2
from pathlib import Path

from tqdm import tqdm

def make_square(img, target_size):
    old_size = img.shape[:2]
    desired_size = max(old_size)
    desired_size = max(desired_size, target_size)

    delta_w = desired_size - old_size[1]
    delta_h = desired_size - old_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [255, 255, 255]
    new_im = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return new_im


def smart_resize(img, size):
    # Assumes the image has already gone through make_square
    if img.shape[0] > size:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    elif img.shape[0] < size:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)
    return img

class Tagger :
    def __init__(self, filename) -> None:
        self.model = InferenceSession(filename, providers=['CUDAExecutionProvider'])
        [root, _] = os.path.split(filename)
        self.tags = pd.read_csv(os.path.join(root, 'selected_tags.csv') if root else 'selected_tags.csv')
        _, self.height, _, _ = self.model.get_inputs()[0].shape

    def label(self, image: Image) -> Dict[str, float] :
        # alpha to white
        image = image.convert('RGBA')
        new_image = Image.new('RGBA', image.size, 'WHITE')
        new_image.paste(image, mask=image)
        image = new_image.convert('RGB')
        image = np.asarray(image)

        # PIL RGB to OpenCV BGR
        image = image[:, :, ::-1]

        image = make_square(image, self.height)
        image = smart_resize(image, self.height)
        image = image.astype(np.float32)
        image = np.expand_dims(image, 0)

        # evaluate model
        input_name = self.model.get_inputs()[0].name
        label_name = self.model.get_outputs()[0].name
        confidents = self.model.run([label_name], {input_name: image})[0]

        tags = self.tags[:][['name']]
        tags['confidents'] = confidents[0]

        # first 4 items are for rating (general, sensitive, questionable, explicit)
        ratings = dict(tags[:4].values)

        # rest are regular tags
        tags = dict(tags[4:].values)

        tags = {t: v for t, v in tags.items() if v > 0.5}
        return tags

    def label_cv2_bgr(self, image: np.ndarray) -> Dict[str, float] :
        # image in BGR u8
        image = make_square(image, self.height)
        image = smart_resize(image, self.height)
        image = image.astype(np.float32)
        image = np.expand_dims(image, 0)

        # evaluate model
        input_name = self.model.get_inputs()[0].name
        label_name = self.model.get_outputs()[0].name
        confidents = self.model.run([label_name], {input_name: image})[0]

        tags = self.tags[:][['name']]
        tags['confidents'] = confidents[0]

        # first 4 items are for rating (general, sensitive, questionable, explicit)
        ratings = dict(tags[:4].values)

        # rest are regular tags
        tags = dict(tags[4:].values)

        tags = {t: v for t, v in tags.items() if v > 0.75}
        return tags
    