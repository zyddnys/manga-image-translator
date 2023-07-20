import os
import torch
import numpy as np
from PIL import Image
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

    async def _infer(self, image: Image.Image, colorization_size: int, apply_denoise=True, denoise_sigma=25) -> Image.Image:
        # Size has to be multiple of 32
        img = np.array(image.convert('RGBA'))
        max_size = min(*img.shape[:2])
        max_size -= max_size % 32
        if colorization_size >= 0:
            size = min(max_size, colorization_size - (colorization_size % 32))
        else:
            # size<=576 gives best results
            size = min(max_size, 576)

        if apply_denoise:
            img = self.denoiser.get_denoised_image(img, sigma=denoise_sigma)

        img, current_pad = resize_pad(img, size)

        transform = ToTensor()
        current_image = transform(img).unsqueeze(0).to(self.device)
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
        return Image.fromarray(colored_image.astype(np.uint8))
