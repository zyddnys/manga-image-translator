import numpy as np
import cv2
import torch
from pathlib import Path
import torch
from typing import Union, Tuple
import einops

from utils import det_rearrange_forward
from .basemodel import TextDetBase, TextDetBaseDNN
from .utils.yolov5_utils import non_max_suppression
from .utils.db_utils import SegDetectorRepresenter
from .utils.imgproc_utils import letterbox
from .textblock import TextBlock, group_output
from .textmask import refine_mask, refine_undetected_mask, REFINEMASK_INPAINT, REFINEMASK_ANNOTATION

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


class TextDetector:
    lang_list = ['eng', 'ja', 'unknown']
    langcls2idx = {'eng': 0, 'ja': 1, 'unknown': 2}

    def __init__(self, model_path, input_size=1024, device='cpu', half=False, nms_thresh=0.35, conf_thresh=0.4, mask_thresh=0.3, act='leaky'):
        super(TextDetector, self).__init__()
        cuda = device == 'cuda'

        if Path(model_path).suffix == '.onnx':
            self.model = cv2.dnn.readNetFromONNX(model_path)
            self.net = TextDetBaseDNN(input_size, model_path)
            self.backend = 'opencv'
        else:
            self.net = TextDetBase(model_path, device=device, act=act)
            self.backend = 'torch'

        if isinstance(input_size, int):
            input_size = (input_size, input_size)
        self.input_size = input_size
        self.device = device
        self.half = half
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.seg_rep = SegDetectorRepresenter(thresh=0.3)

    def det_batch_forward_ctd(self, batch: np.ndarray, device: str) -> Tuple[np.ndarray, np.ndarray]:
        
        if isinstance(self.net, TextDetBase):
            batch = einops.rearrange(batch.astype(np.float32) / 255., 'n h w c -> n c h w')
            batch = torch.from_numpy(batch).to(device)
            _, mask, lines = self.net(batch)
            mask = mask.cpu().numpy()
            lines = lines.cpu().numpy()
        elif isinstance(self.net, TextDetBaseDNN):
            mask_lst, line_lst = [], []
            for b in batch:
                _, mask, lines = self.net(b)
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
    def __call__(self, img, refine_mode=REFINEMASK_INPAINT, keep_undetected_mask=False, bgr2rgb=True, verbose=False, det_rearrange_max_batches=4):
        im_h, im_w = img.shape[:2]
        lines_map, mask = det_rearrange_forward(img, self.det_batch_forward_ctd, self.input_size[0], det_rearrange_max_batches, self.device, verbose)
        blks = []
        resize_ratio = [1, 1]
        if lines_map is None:
            img_in, ratio, dw, dh = preprocess_img(img, input_size=self.input_size, device=self.device, half=self.half, to_tensor=self.backend=='torch')
            blks, mask, lines_map = self.net(img_in)

            if self.backend == 'opencv':
                if mask.shape[1] == 2:     # some version of opencv spit out reversed result
                    tmp = mask
                    mask = lines_map
                    lines_map = tmp
            mask = mask.squeeze()
            resize_ratio = (im_w / (self.input_size[0] - dw), im_h / (self.input_size[1] - dh))
            blks = postprocess_yolo(blks, self.conf_thresh, self.nms_thresh, resize_ratio)
            mask = mask[..., :mask.shape[0]-dh, :mask.shape[1]-dw]
            lines_map = lines_map[..., :lines_map.shape[2]-dh, :lines_map.shape[3]-dw]

        mask = postprocess_mask(mask)
        lines, scores = self.seg_rep(None, lines_map, height=im_h, width=im_w)
        box_thresh = 0.6
        idx = np.where(scores[0] > box_thresh)
        lines, scores = lines[0][idx], scores[0][idx]

        # map output to input img
        mask = cv2.resize(mask, (im_w, im_h), interpolation=cv2.INTER_LINEAR)
        if lines.size == 0:
            lines = []
        else:
            lines = lines.astype(np.int32)
        blk_list = group_output(blks, lines, im_w, im_h, mask)
        mask_refined = refine_mask(img, mask, blk_list, refine_mode=refine_mode)
        if keep_undetected_mask:
            mask_refined = refine_undetected_mask(img, mask, mask_refined, blk_list, refine_mode=refine_mode)

        return mask, mask_refined, blk_list

DEFAULT_MODEL = None
def load_model(cuda: bool):
    global DEFAULT_MODEL
    device = 'cuda' if cuda else 'cpu'
    if cuda:
        model = TextDetector(model_path='comictextdetector.pt', device=device, act='leaky')
        model.net.cuda()
    else:
        model = TextDetector(model_path='comictextdetector.pt.onnx', device=device, act='leaky', input_size=1024)
    DEFAULT_MODEL = model

async def dispatch(img: np.ndarray, cuda: bool, verbose: bool = False, det_rearrange_max_batches: int = 4):
    global DEFAULT_MODEL
    if DEFAULT_MODEL is None:
        load_model(cuda)
    return DEFAULT_MODEL(img, refine_mode=REFINEMASK_INPAINT, keep_undetected_mask=False, bgr2rgb=False, verbose=verbose, det_rearrange_max_batches=det_rearrange_max_batches)