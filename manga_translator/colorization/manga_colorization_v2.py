import os

import cv2
import torch
import numpy as np
from torchvision.transforms import ToTensor

from .common import OfflineColorizer
from .manga_colorization_v2_utils.networks.models import Colorizer
from .manga_colorization_v2_utils.denoising.denoiser import FFDNetDenoiser
from .manga_colorization_v2_utils.utils.utils import resize_pad


# https://github.com/qweasdd/manga-colorization-v2
class MangaColorizationV2(OfflineColorizer):
    _MODEL_SUB_DIR = os.path.join(OfflineColorizer._MODEL_SUB_DIR, 'manga-colorization-v2')
    _MODEL_MAPPING = {
        # Models were in google drive so had to upload to github
        'generator': {
            'url': 'https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/manga-colorization-v2-generator.zip',
            'file': 'generator.zip',
            'hash': '087e6a0bc02770e732a52f33878b71a272a6123c9ac649e9b5bfb75e39e5c1d5',
        },
        'denoiser': {
            'url': 'https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/manga-colorization-v2-net_rgb.pth',
            'file': 'net_rgb.pth',
            'hash': '0fe98bfd2ac870b15f360661b1c4789eecefc6dc2e4462842a0dd15e149a0433',
        }
    }

    async def _load(self, device: str):
        self.device = device
        self.colorizer = Colorizer().to(device)
        self.colorizer.generator.load_state_dict(
            torch.load(self._get_file_path('generator.zip'), map_location=self.device))
        self.colorizer = self.colorizer.eval()
        self.denoiser = FFDNetDenoiser(device, _weights_dir=self.model_dir)

    async def _unload(self):
        del self.colorizer
        del self.denoiser

    async def _infer(self, image: np.ndarray, apply_denoise=True, denoise_sigma=25) -> np.ndarray:
        # Size has to be multiple of 32
        image_dim = image.shape
        image_width = image.shape[1]
        extra = get_extra(image_width)
        size = image_width + extra
        # TODO: Add automatic upscaling to approximate original size

        if apply_denoise:
            image = self.denoiser.get_denoised_image(image, sigma=denoise_sigma)

        image, current_pad = resize_pad(image, size)
        transform = ToTensor()
        current_image = transform(image).unsqueeze(0).to(self.device)
        current_hint = torch.zeros(1, 4, current_image.shape[2], current_image.shape[3]).float().to(self.device)

        with torch.no_grad():
            fake_color, _ = self.colorizer(torch.cat([current_image, current_hint], 1))
            fake_color = fake_color.detach()

        result = fake_color[0].detach().cpu().permute(1, 2, 0) * 0.5 + 0.5

        if current_pad[0] != 0:
            result = result[:-current_pad[0]]
        if current_pad[1] != 0:
            result = result[:, :-current_pad[1]]

        colored_image = result.numpy() * 255
        return cv2.resize(colored_image, (image_dim[1], image_dim[0]), interpolation=cv2.INTER_CUBIC)


def get_extra(width: int):
    return 32 - (width % 32)
