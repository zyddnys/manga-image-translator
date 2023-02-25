import os
import re
import subprocess
import tempfile
import shutil
import tqdm
from sys import platform
from typing import List
from PIL import Image

from .common import OfflineUpscaler

if platform == 'win32':
    esrgan_base_folder = 'esrgan-win/'
    esrgan_executable_path = os.path.join(esrgan_base_folder, 'realesrgan-ncnn-vulkan.exe')
    model_mapping = {
        'esrgan-win': {
            'url': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-windows.zip',
            'hash': 'abc02804e17982a3be33675e4d471e91ea374e65b70167abc09e31acb412802d',
            'archive': {
                '*': esrgan_base_folder,
            },
        },
    }
elif platform == 'darwin':
    esrgan_base_folder = 'esrgan-macos/'
    esrgan_executable_path = os.path.join(esrgan_base_folder, 'realesrgan-ncnn-vulkan')
    model_mapping = {
        'esrgan-macos': {
            'url': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-macos.zip',
            'hash': 'e0ad05580abfeb25f8d8fb55aaf7bedf552c375b5b4d9bd3c8d59764d2cc333a',
            'archive': {
                '*': esrgan_base_folder,
            },
        },
    }
else:
    esrgan_base_folder = 'esrgan-linux/'
    esrgan_executable_path = os.path.join(esrgan_base_folder, 'realesrgan-ncnn-vulkan')
    model_mapping = {
        'esrgan-linux': {
            'url': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-ubuntu.zip',
            'hash': 'e5aa6eb131234b87c0c51f82b89390f5e3e642b7b70f2b9bbe95b6a285a40c96',
            'archive': {
                '*': esrgan_base_folder,
            },
            'executables': [
                esrgan_executable_path
            ],
        },
    }

# https://github.com/xinntao/Real-ESRGAN
class ESRGANUpscaler(OfflineUpscaler):
    _MODEL_MAPPING = model_mapping
    _VALID_UPSCALE_RATIOS = [2, 3, 4]

    async def _load(self, device: str):
        pass

    async def _unload(self):
        pass

    async def _forward(self, image_batch: List[Image.Image], upscale_ratio: float) -> List[Image.Image]:
        # Has to cache images because chosen upscaler doesn't support piping
        in_dir = tempfile.mkdtemp()
        out_dir = tempfile.mkdtemp()
        for i, image in enumerate(image_batch):
            image.save(os.path.join(in_dir, f'{i}.png'))

        try:
            self._run_esrgan_executable(in_dir, out_dir, upscale_ratio, 0)
        except Exception:
            # Maybe throw exception instead
            self.logger.warn(f'Process returned non-zero exit status. Skipping upscaling.')
            return image_batch

        output_batch = []
        for i, image in enumerate(image_batch):
            img_path = os.path.join(out_dir, f'{i}.png')
            if os.path.exists(img_path):
                img = Image.open(img_path)
                img.load()
                output_batch.append(img)
            else:
                output_batch.append(image)

        shutil.rmtree(in_dir)
        shutil.rmtree(out_dir)
        return output_batch

    def _run_esrgan_executable(self, image_directory: str, output_directory: str, upscale_ratio: float, denoise_level: int):
        cmds = [
            self._get_file_path(esrgan_executable_path),
            '-i', image_directory,
            '-o', output_directory,
            '-m', self._get_file_path(os.path.join(esrgan_base_folder, 'models')),
            '-s', str(upscale_ratio),
        ]
        process = subprocess.Popen(cmds, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        with tqdm.tqdm(desc='[esgran]', total=100) as bar:
            last_progress = 0
            for line in iter(process.stdout.readline, b''):
                match = re.search(r'^(\d+\.\d+)%$', str(line, 'utf-8'))
                if match:
                    progress = float(match.group(1))
                    bar.update(progress - last_progress)
                    last_progress = progress
            bar.update(100 - last_progress)
