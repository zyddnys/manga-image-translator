from PIL import Image
from manga_translator.utils import load_image
import numpy as np

def load_rgb_image(path: str) -> np.ndarray:
    img_rgb, _ = load_image(Image.open(path))
    return img_rgb