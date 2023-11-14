from PIL import Image
from typing import List
from abc import abstractmethod

from ..utils import InfererModule, ModelWrapper

class CommonUpscaler(InfererModule):
    _VALID_UPSCALE_RATIOS = []

    async def upscale(self, image_batch: List[Image.Image], upscale_ratio: float) -> List[Image.Image]:
        if upscale_ratio == 1:
            return image_batch

        self._VALID_UPSCALE_RATIOS.sort()
        assert(self._VALID_UPSCALE_RATIOS[0] > 1)

        ratio_left = upscale_ratio
        while ratio_left > 0:
            ratio = self._VALID_UPSCALE_RATIOS[-1]
            for valid_ratio in self._VALID_UPSCALE_RATIOS:
                if ratio_left <= valid_ratio:
                    ratio = valid_ratio
                    break
            ratio_left -= ratio
            if upscale_ratio > self._VALID_UPSCALE_RATIOS[-1]:
                self.logger.info(f'Upscaling image by {ratio}; left: {ratio_left}')
            image_batch = await self._upscale(image_batch, ratio)
        if ratio_left < 0:
            downscale_ratio = (ratio + ratio_left) / ratio
            self.logger.info(f'Downscaling image by {downscale_ratio} to correct upscale ratio')
            for i, image in enumerate(image_batch):
                image_batch[i] = image.resize((int(image.size[0] * downscale_ratio), int(image.size[1] * downscale_ratio)))
        return image_batch

    @abstractmethod
    async def _upscale(self, image_batch: List[Image.Image], upscale_ratio: float) -> List[Image.Image]:
        pass

class OfflineUpscaler(CommonUpscaler, ModelWrapper):
    _MODEL_SUB_DIR = 'upscaling'

    async def _upscale(self, *args, **kwargs):
        return await self.infer(*args, **kwargs)

    @abstractmethod
    async def _infer(self, image_batch: List[Image.Image], upscale_ratio: float) -> List[Image.Image]:
        """
        Perform the actual upscaling of the images.

        Args:
            image_batch: The list of images to upscale.
            upscale_ratio: The upscale ratio to use.

        Returns:
            The list of upscaled images.
        """
        pass
