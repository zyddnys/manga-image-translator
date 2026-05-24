import os
import shutil
import numpy as np
import einops
from typing import Union, Tuple
import cv2
import torch

from .ctd_utils.basemodel import TextDetBase, TextDetBaseDNN
from .ctd_utils.utils.yolov5_utils import non_max_suppression
from .ctd_utils.utils.db_utils import SegDetectorRepresenter
from .ctd_utils.utils.imgproc_utils import letterbox
from .ctd_utils.textmask import REFINEMASK_INPAINT, refine_mask
from .common import OfflineDetector
from ..utils import Quadrilateral, det_rearrange_forward

def preprocess_img(img, input_size=(1024, 1024), device='cpu', bgr2rgb=True, half=False, to_tensor=True):
    if bgr2rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_in, ratio, (dw, dh) = letterbox(img, new_shape=input_size, auto=False, stride=64)
    if to_tensor:
        img_in = img_in.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img_in = np.array([np.ascontiguousarray(img_in)]).astype(np.float32) / 255
        if to_tensor:
            img_in = torch.from_numpy(img_in).to(device)
            if half:
                img_in = img_in.half()
    return img_in, ratio, int(dw), int(dh)

def postprocess_mask(img: Union[torch.Tensor, np.ndarray], thresh=None):
    # img = img.permute(1, 2, 0)
    if isinstance(img, torch.Tensor):
        img = img.squeeze_()
        if img.device != 'cpu':
            img = img.detach().cpu()
        img = img.numpy()
    else:
        img = img.squeeze()
    if thresh is not None:
        img = img > thresh
    img = img * 255
    # if isinstance(img, torch.Tensor):

    return img.astype(np.uint8)

def postprocess_yolo(det, conf_thresh, nms_thresh, resize_ratio, sort_func=None):
    det = non_max_suppression(det, conf_thresh, nms_thresh)[0]
    # bbox = det[..., 0:4]
    if det.device != 'cpu':
        det = det.detach_().cpu().numpy()
    det[..., [0, 2]] = det[..., [0, 2]] * resize_ratio[0]
    det[..., [1, 3]] = det[..., [1, 3]] * resize_ratio[1]
    if sort_func is not None:
        det = sort_func(det)

    blines = det[..., 0:4].astype(np.int32)
    confs = np.round(det[..., 4], 3)
    cls = det[..., 5].astype(np.int32)
    return blines, cls, confs


class ComicTextDetector(OfflineDetector):
    _MODEL_MAPPING = {
        'model-cuda': {
            'url': 'https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/comictextdetector.pt',
            'hash': '1f90fa60aeeb1eb82e2ac1167a66bf139a8a61b8780acd351ead55268540cccb',
            'file': '.',
        },
        'model-cpu': {
            'url': 'https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/comictextdetector.pt.onnx',
            'hash': '1a86ace74961413cbd650002e7bb4dcec4980ffa21b2f19b86933372071d718f',
            'file': '.',
        },
    }

    def __init__(self, *args, **kwargs):
        os.makedirs(self.model_dir, exist_ok=True)
        if os.path.exists('comictextdetector.pt'):
            shutil.move('comictextdetector.pt', self._get_file_path('comictextdetector.pt'))
        if os.path.exists('comictextdetector.pt.onnx'):
            shutil.move('comictextdetector.pt.onnx', self._get_file_path('comictextdetector.pt.onnx'))
        super().__init__(*args, **kwargs)

    async def _load(self, device: str, input_size=1024, half=False, nms_thresh=0.35, conf_thresh=0.4):
        self.device = device
        if self.device == 'cuda' or self.device == 'mps':
            self.model = TextDetBase(self._get_file_path('comictextdetector.pt'), device=self.device, act='leaky')
            self.model.to(self.device)
            self.backend = 'torch'
        else:
            model_path = self._get_file_path('comictextdetector.pt.onnx')
            self.model = cv2.dnn.readNetFromONNX(model_path)
            self.model = TextDetBaseDNN(input_size, model_path)
            self.backend = 'opencv'

        if isinstance(input_size, int):
            input_size = (input_size, input_size)
        self.input_size = input_size
        self.half = half
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.seg_rep = SegDetectorRepresenter(thresh=0.3)

    async def _unload(self):
        del self.model

    @staticmethod
    def _normalize_detect_size(detect_size: int, fallback: int = 1024) -> int:
        """
        CTD is fully-convolutional in the torch backend, so it can benefit from
        higher detect sizes for tiny/SFX lettering. Keep the size on YOLO/UNet's
        stride boundary and cap it to avoid accidental VRAM explosions.
        """
        try:
            size = int(detect_size)
        except Exception:
            size = fallback
        size = max(512, min(3072, size))
        return int(((size + 63) // 64) * 64)

    @staticmethod
    def _ctd_text_thresh(text_threshold: float) -> float:
        """
        MIT's global default is 0.5, but CTD historically used 0.3 internally.
        Treat values <= 0.5 as "sensitive CTD mode" so UI values such as 0.4
        do not accidentally become stricter than the old hard-coded behavior.
        """
        try:
            thresh = float(text_threshold)
        except Exception:
            thresh = 0.5
        if thresh <= 0.5:
            thresh = min(0.3, thresh)
        return max(0.05, min(0.95, thresh))

    @staticmethod
    def _ctd_box_thresh(box_threshold: float) -> float:
        """
        CTD used to ignore MIT's box_threshold and hard-code 0.6. Preserve that
        when config is left at MIT default 0.7, but honor lower user values for
        SFX/small text detection.
        """
        try:
            thresh = float(box_threshold)
        except Exception:
            thresh = 0.7
        if abs(thresh - 0.7) < 1e-6:
            thresh = 0.6
        return max(0.05, min(0.95, thresh))

    @staticmethod
    def _ctd_unclip_ratio(unclip_ratio: float) -> float:
        try:
            ratio = float(unclip_ratio)
        except Exception:
            ratio = 1.5
        return max(0.5, min(6.0, ratio))

    def det_batch_forward_ctd(self, batch: np.ndarray, device: str) -> Tuple[np.ndarray, np.ndarray]:
        if isinstance(self.model, TextDetBase):
            batch = einops.rearrange(batch.astype(np.float32) / 255., 'n h w c -> n c h w')
            batch = torch.from_numpy(batch).to(device)
            _, mask, lines = self.model(batch)
            mask = mask.detach().cpu().numpy()
            lines = lines.detach().cpu().numpy()
        elif isinstance(self.model, TextDetBaseDNN):
            mask_lst, line_lst = [], []
            for b in batch:
                _, mask, lines = self.model(b)
                if mask.shape[1] == 2:     # some version of opencv spit out reversed result
                    tmp = mask
                    mask = lines
                    lines = tmp
                mask_lst.append(mask)
                line_lst.append(lines)
            lines, mask = np.concatenate(line_lst, 0), np.concatenate(mask_lst, 0)
        else:
            raise NotImplementedError
        return lines, mask

    @torch.no_grad()
    async def _infer(self, image: np.ndarray, detect_size: int, text_threshold: float, box_threshold: float,
                     unclip_ratio: float, verbose: bool = False):

        # keep_undetected_mask = False
        # refine_mode = REFINEMASK_INPAINT

        im_h, im_w = image.shape[:2]
        runtime_size = self._normalize_detect_size(detect_size, self.input_size[0])
        # The ONNX/OpenCV backend is exported for a fixed input size. Only the
        # torch backend should use user-selected high-resolution detection.
        if self.backend != 'torch':
            runtime_size = self.input_size[0]

        seg_thresh = self._ctd_text_thresh(text_threshold)
        score_thresh = self._ctd_box_thresh(box_threshold)
        ctd_unclip_ratio = self._ctd_unclip_ratio(unclip_ratio)

        if verbose:
            self.logger.info(
                f'CTD config: detect_size={runtime_size}, '
                f'text_threshold={seg_thresh:.2f}, '
                f'box_threshold={score_thresh:.2f}, '
                f'unclip_ratio={ctd_unclip_ratio:.2f}'
            )

        lines_map, mask = det_rearrange_forward(image, self.det_batch_forward_ctd, runtime_size, 4, self.device, verbose)
        # blks = []
        # resize_ratio = [1, 1]
        if lines_map is None:
            runtime_input_size = (runtime_size, runtime_size) if self.backend == 'torch' else self.input_size
            img_in, ratio, dw, dh = preprocess_img(image, input_size=runtime_input_size, device=self.device, half=self.half, to_tensor=self.backend=='torch')
            blks, mask, lines_map = self.model(img_in)

            if self.backend == 'opencv':
                if mask.shape[1] == 2: # some version of opencv spit out reversed result
                    tmp = mask
                    mask = lines_map
                    lines_map = tmp
            mask = mask.squeeze()
            # resize_ratio = (im_w / (self.input_size[0] - dw), im_h / (self.input_size[1] - dh))
            # blks = postprocess_yolo(blks, self.conf_thresh, self.nms_thresh, resize_ratio)
            mask = mask[..., :mask.shape[0]-dh, :mask.shape[1]-dw]
            lines_map = lines_map[..., :lines_map.shape[2]-dh, :lines_map.shape[3]-dw]

        mask = postprocess_mask(mask)
        # Recreate SegDetectorRepresenter per request so UI thresholds actually
        # affect CTD. Previously CTD ignored text_threshold/unclip_ratio and
        # hard-coded a 0.6 score cutoff, causing tiny SFX/handwritten text to be
        # dropped even when the UI was set to aggressive values.
        seg_rep = SegDetectorRepresenter(
            thresh=seg_thresh,
            box_thresh=score_thresh,
            unclip_ratio=ctd_unclip_ratio,
        )
        lines, scores = seg_rep(None, lines_map, height=im_h, width=im_w)
        idx = np.where(scores[0] > score_thresh)
        lines, scores = lines[0][idx], scores[0][idx]

        # map output to input img
        mask = cv2.resize(mask, (im_w, im_h), interpolation=cv2.INTER_LINEAR)

        # if lines.size == 0:
        #     lines = []
        # else:
        #     lines = lines.astype(np.int32)

        # YOLO was used for finding bboxes which to order the lines into. This is now solved
        # through the textline merger, which seems to work more reliably.
        # The YOLO language detection seems unnecessary as it could never be as good as
        # using the OCR extracted string directly.
        # Doing it for increasing the textline merge accuracy doesn't really work either,
        # as the merge could be postponed until after the OCR finishes.

        textlines = [Quadrilateral(pts.astype(int), '', score) for pts, score in zip(lines, scores)]
        mask_refined = refine_mask(image, mask, textlines, refine_mode=None)

        return textlines, mask_refined, None

        # blk_list = group_output(blks, lines, im_w, im_h, mask)
        # mask_refined = refine_mask(image, mask, blk_list, refine_mode=refine_mode)
        # if keep_undetected_mask:
        #     mask_refined = refine_undetected_mask(image, mask, mask_refined, blk_list, refine_mode=refine_mode)

        # return blk_list, mask, mask_refined
