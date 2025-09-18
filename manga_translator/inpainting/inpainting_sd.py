import torch
import numpy as np
import cv2
import os
import einops
import safetensors
import safetensors.torch
from PIL import Image
from omegaconf import OmegaConf

from .common import OfflineInpainter
from ..config import InpainterConfig
from ..utils import resize_keep_aspect

from .booru_tagger import Tagger
from .sd_hack import hack_everything
from .ldm.util import instantiate_from_config


def get_state_dict(d):
    return d.get('state_dict', d)

def load_state_dict(ckpt_path, location='cpu'):
    _, extension = os.path.splitext(ckpt_path)
    if extension.lower() == ".safetensors":
        import safetensors.torch
        state_dict = safetensors.torch.load_file(ckpt_path, device=location)
    else:
        state_dict = get_state_dict(torch.load(ckpt_path, map_location=torch.device(location)))
    state_dict = get_state_dict(state_dict)
    return state_dict


def create_model(config_path):
    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model).cpu()
    return model


def load_ldm_sd(model, path) :
    if path.endswith('.safetensor') :
        sd = safetensors.torch.load_file(path)
    else :
        sd = load_state_dict(path)
    model.load_state_dict(sd, strict = False)

class StableDiffusionInpainter(OfflineInpainter):
    _MODEL_MAPPING = {
        'model_grapefruit': {
            'url': 'https://civitai.com/api/download/models/8364',
            'hash': 'dd680bd77d553e095faf58ff8c12584efe2a9b844e18bcc6ba2a366b85caceb8',
            'file': 'abyssorangemix2_Hard-inpainting.safetensors',
        },
        'model_wd_swinv2': {
            'url': 'https://huggingface.co/SmilingWolf/wd-v1-4-swinv2-tagger-v2/resolve/main/model.onnx',
            'hash': '04ec04fdf7db74b4fed7f4b52f52e04dec4dbad9e4d88d2d178f334079a29fde',
            'file': 'wd_swinv2.onnx',
        },
        'model_wd_swinv2_csv': {
            'url': 'https://huggingface.co/SmilingWolf/wd-v1-4-swinv2-tagger-v2/raw/main/selected_tags.csv',
            'hash': '8c8750600db36233a1b274ac88bd46289e588b338218c2e4c62bbc9f2b516368',
            'file': 'selected_tags.csv',
        }
    }

    def __init__(self, *args, **kwargs):
        os.makedirs(self.model_dir, exist_ok=True)
        super().__init__(*args, **kwargs)

    async def _load(self, device: str):
        self.tagger = Tagger(self._get_file_path('wd_swinv2.onnx'))
        self.model = create_model('manga_translator/inpainting/guided_ldm_inpaint9_v15.yaml').cuda()
        load_ldm_sd(self.model, self._get_file_path('abyssorangemix2_Hard-inpainting.safetensors'))
        hack_everything()
        self.model.eval()
        self.device = device
        self.model = self.model.to(device)

    async def _unload(self):
        del self.model

    @torch.no_grad()
    async def _infer(self, image: np.ndarray, mask: np.ndarray, config: InpainterConfig, inpainting_size: int = 1024, verbose: bool = False) -> np.ndarray:
        img_original = np.copy(image)
        mask_original = np.copy(mask)
        mask_original[mask_original < 127] = 0
        mask_original[mask_original >= 127] = 1
        mask_original = mask_original[:, :, None]

        height, width, c = image.shape
        if max(image.shape[0: 2]) > inpainting_size:
            image = resize_keep_aspect(image, inpainting_size)
            mask = resize_keep_aspect(mask, inpainting_size)
        pad_size = 64
        h, w, c = image.shape
        if h % pad_size != 0:
            new_h = (pad_size - (h % pad_size)) + h
        else:
            new_h = h
        if w % pad_size != 0:
            new_w = (pad_size - (w % pad_size)) + w
        else:
            new_w = w
        if new_h != h or new_w != w:
            image = cv2.resize(image, (new_w, new_h), interpolation = cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (new_w, new_h), interpolation = cv2.INTER_LINEAR)
        self.logger.info(f'Inpainting resolution: {new_w}x{new_h}')
        tags = self.tagger.label_cv2_bgr(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        self.logger.info(f'tags={list(tags.keys())}')
        blacklist = set()
        pos_prompt = ','.join([x for x in tags.keys() if x not in blacklist]).replace('_', ' ')
        pos_prompt = 'masterpiece,best quality,' + pos_prompt
        neg_prompt = 'worst quality, low quality, normal quality,text,text,text,text'
        if self.device.startswith('cuda') :
            with torch.autocast(enabled = True, device_type = 'cuda') :
                img = self.model.img2img_inpaint(
                    image = Image.fromarray(image),
                    c_text = pos_prompt,
                    uc_text = neg_prompt,
                    mask = Image.fromarray(mask),
                    device = self.device
                    )
        else :
            img = self.model.img2img_inpaint(
                image = Image.fromarray(image),
                c_text = pos_prompt,
                uc_text = neg_prompt,
                mask = Image.fromarray(mask),
                device = self.device
                )

        img_inpainted = (einops.rearrange(img, '1 c h w -> h w c').cpu().numpy() * 127.5 + 127.5).astype(np.uint8)
        if new_h != height or new_w != width:
            img_inpainted = cv2.resize(img_inpainted, (width, height), interpolation = cv2.INTER_LINEAR)
        ans = img_inpainted * mask_original + img_original * (1 - mask_original)
        return ans
