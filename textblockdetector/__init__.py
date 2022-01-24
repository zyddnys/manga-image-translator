import json
from .basemodel import TextDetBase
import os.path as osp
from tqdm import tqdm
import numpy as np
import cv2
import torch
from pathlib import Path
import torch
import onnxruntime
from .utils.yolov5_utils import non_max_suppression
from .utils.db_utils import SegDetectorRepresenter
from .utils.io_utils import imread, imwrite, find_all_imgs, NumpyEncoder
from .utils.imgproc_utils import letterbox, xyxy2yolo, get_yololabel_strings
from .textblock import TextBlock, group_output
from .textmask import refine_mask, refine_undetected_mask, REFINEMASK_INPAINT, REFINEMASK_ANNOTATION

def preprocess_img(img, input_size=(1024, 1024), device='cpu', bgr2rgb=True, half=False, to_tensor=True):
    if bgr2rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_in, ratio, (dw, dh) = letterbox(img, new_shape=input_size, auto=False, stride=64)
    img_in = img_in.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img_in = np.array([np.ascontiguousarray(img_in)]).astype(np.float32) / 255
    if to_tensor:
        img_in = torch.from_numpy(img_in).to(device)
        if half:
            img_in = img_in.half()
    return img_in, ratio, int(dw), int(dh)

def postprocess_mask(img: torch.Tensor, thresh=None):
    # img = img.permute(1, 2, 0)
    if thresh is not None:
        img = img > thresh
    img = img * 255
    if img.device != 'cpu':
        img = img.detach_().cpu()
    img = img.numpy().astype(np.uint8)
    return img

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

    def __init__(self, model_path, input_size=1152, device='cpu', half=False, nms_thresh=0.35, conf_thresh=0.4, mask_thresh=0.3, act='leaky', backend='torch') :
        super(TextDetector, self).__init__()
        cuda = device == 'cuda'
        self.backend = backend
        if self.backend == 'torch':
            self.net = TextDetBase(model_path, device=device, act=act)
        else:
            # TODO: OPENCV ONNX INFERENCE
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
            self.session = onnxruntime.InferenceSession(model_path, providers=providers)
        if isinstance(input_size, int):
            input_size = (input_size, input_size)
        self.input_size = input_size
        self.device = device
        self.half = half
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.seg_rep = SegDetectorRepresenter(thresh=0.3)

    def __call__(self, img, refine_mode=REFINEMASK_INPAINT, keep_undetected_mask=False, bgr2rgb=True):
        img_in, ratio, dw, dh = preprocess_img(img, input_size=self.input_size, device=self.device, half=self.half, bgr2rgb=bgr2rgb)

        im_h, im_w = img.shape[:2]
        with torch.no_grad():
            blks, mask, lines_map = self.net(img_in)
        
        resize_ratio = (im_w / (self.input_size[0] - dw), im_h / (self.input_size[1] - dh))
        blks = postprocess_yolo(blks[0], self.conf_thresh, self.nms_thresh, resize_ratio)
        mask = postprocess_mask(mask.squeeze_())
        lines, scores = self.seg_rep(self.input_size, lines_map)
        box_thresh = 0.6
        idx = np.where(scores[0] > box_thresh)
        lines, scores = lines[0][idx], scores[0][idx]
        
        # map output to input img
        mask = mask[: mask.shape[0]-dh, : mask.shape[1]-dw]
        mask = cv2.resize(mask, (im_w, im_h), interpolation=cv2.INTER_LINEAR)
        if lines.size == 0 :
            lines = []
        else :
            lines = lines.astype(np.float64)
            lines[..., 0] *= resize_ratio[0]
            lines[..., 1] *= resize_ratio[1]
            lines = lines.astype(np.int32)
        blk_list = group_output(blks, lines, im_w, im_h, mask)
        mask_refined = refine_mask(img, mask, blk_list, refine_mode=refine_mode)
        if keep_undetected_mask:
            mask_refined = refine_undetected_mask(img, mask, mask_refined, blk_list, refine_mode=refine_mode)
    
        return mask, mask_refined, blk_list

    def cuda(self):
        self.net.to('cuda')

DEFAULT_MODEL = None
def load_model(cuda: bool):
    global DEFAULT_MODEL
    device = 'cuda' if cuda else 'cpu'
    model = TextDetector(model_path='comictextdetector.pt', device=device, act='leaky')
    if cuda :
        model.cuda()
    DEFAULT_MODEL = model

async def dispatch(img: np.ndarray, cuda: bool):
    global DEFAULT_MODEL
    if DEFAULT_MODEL is None :
        load_model(cuda)
    return DEFAULT_MODEL(img, refine_mode=REFINEMASK_INPAINT, keep_undetected_mask=False, bgr2rgb=False)