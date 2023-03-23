import os
import shutil
import numpy as np
import einops
from typing import List, Union, Tuple
import cv2
import torch

from .ctd_utils.basemodel import TextDetBase, TextDetBaseDNN
from .ctd_utils.utils.yolov5_utils import non_max_suppression
from .ctd_utils.utils.db_utils import SegDetectorRepresenter
from .ctd_utils.utils.imgproc_utils import letterbox
from .ctd_utils.textmask import REFINEMASK_INPAINT, refine_mask
from .common import OfflineDetector
from ..utils import TextBlock, Quadrilateral, det_rearrange_forward

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
        if self.device == 'cuda':
            self.model = TextDetBase(self._get_file_path('comictextdetector.pt'), device=self.device, act='leaky')
            self.model.cuda()
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

    def det_batch_forward_ctd(self, batch: np.ndarray, device: str) -> Tuple[np.ndarray, np.ndarray]:
        if isinstance(self.model, TextDetBase):
            batch = einops.rearrange(batch.astype(np.float32) / 255., 'n h w c -> n c h w')
            batch = torch.from_numpy(batch).to(device)
            _, mask, lines = self.model(batch)
            mask = mask.cpu().numpy()
            lines = lines.cpu().numpy()
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
                     unclip_ratio: float, verbose: bool = False) -> Tuple[List[TextBlock], np.ndarray]:

        # keep_undetected_mask = False
        refine_mode = REFINEMASK_INPAINT

        im_h, im_w = image.shape[:2]
        lines_map, mask = det_rearrange_forward(image, self.det_batch_forward_ctd, self.input_size[0], 4, self.device, verbose)
        # blks = []
        # resize_ratio = [1, 1]
        if lines_map is None:
            img_in, ratio, dw, dh = preprocess_img(image, input_size=self.input_size, device=self.device, half=self.half, to_tensor=self.backend=='torch')
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
        lines, scores = self.seg_rep(None, lines_map, height=im_h, width=im_w)
        box_thresh = 0.6
        idx = np.where(scores[0] > box_thresh)
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
        text_regions = await self._merge_textlines(textlines, image.shape[1], image.shape[0], verbose=verbose)
        mask_refined = refine_mask(image, mask, text_regions, refine_mode=refine_mode)

        return text_regions, mask, mask_refined

        # blk_list = group_output(blks, lines, im_w, im_h, mask)
        # mask_refined = refine_mask(image, mask, blk_list, refine_mode=refine_mode)
        # if keep_undetected_mask:
        #     mask_refined = refine_undetected_mask(image, mask, mask_refined, blk_list, refine_mode=refine_mode)

        # return blk_list, mask, mask_refined

    def _sort_regions(self, regions: List[TextBlock], width: int, height: int) -> List[TextBlock]:
        # Sort regions from left to right, top to bottom
        sorted_regions = []
        for region in sorted(regions, key=lambda region: region.center[1]):
            for i, sorted_region in enumerate(sorted_regions):
                if region.center[1] > sorted_region.xyxy[3]:
                    continue
                if region.center[1] < sorted_region.xyxy[1]:
                    sorted_regions.insert(i + 1, region)
                    break
                # y center of region inside sorted_region so sort by x instead
                if region.center[0] < sorted_region.center[0]:
                    sorted_regions.insert(i, region)
                    break
            else:
                sorted_regions.append(region)
        return sorted_regions
