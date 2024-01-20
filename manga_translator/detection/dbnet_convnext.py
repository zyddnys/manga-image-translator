
from functools import partial
import shutil
from typing import Callable, Optional, Tuple, Union
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torchvision.models import resnet34

import einops
import math

from timm.layers import trunc_normal_, AvgPool2dSame, DropPath, Mlp, GlobalResponseNormMlp, \
	LayerNorm2d, LayerNorm, create_conv2d, get_act_layer, make_divisible, to_ntuple

class Downsample(nn.Module):

	def __init__(self, in_chs, out_chs, stride=1, dilation=1):
		super().__init__()
		avg_stride = stride if dilation == 1 else 1
		if stride > 1 or dilation > 1:
			avg_pool_fn = AvgPool2dSame if avg_stride == 1 and dilation > 1 else nn.AvgPool2d
			self.pool = avg_pool_fn(2, avg_stride, ceil_mode=True, count_include_pad=False)
		else:
			self.pool = nn.Identity()

		if in_chs != out_chs:
			self.conv = create_conv2d(in_chs, out_chs, 1, stride=1)
		else:
			self.conv = nn.Identity()

	def forward(self, x):
		x = self.pool(x)
		x = self.conv(x)
		return x


class ConvNeXtBlock(nn.Module):
	""" ConvNeXt Block
	There are two equivalent implementations:
	  (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
	  (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back

	Unlike the official impl, this one allows choice of 1 or 2, 1x1 conv can be faster with appropriate
	choice of LayerNorm impl, however as model size increases the tradeoffs appear to change and nn.Linear
	is a better choice. This was observed with PyTorch 1.10 on 3090 GPU, it could change over time & w/ different HW.
	"""

	def __init__(
			self,
			in_chs: int,
			out_chs: Optional[int] = None,
			kernel_size: int = 7,
			stride: int = 1,
			dilation: Union[int, Tuple[int, int]] = (1, 1),
			mlp_ratio: float = 4,
			conv_mlp: bool = False,
			conv_bias: bool = True,
			use_grn: bool = False,
			ls_init_value: Optional[float] = 1e-6,
			act_layer: Union[str, Callable] = 'gelu',
			norm_layer: Optional[Callable] = None,
			drop_path: float = 0.,
	):
		"""

		Args:
			in_chs: Block input channels.
			out_chs: Block output channels (same as in_chs if None).
			kernel_size: Depthwise convolution kernel size.
			stride: Stride of depthwise convolution.
			dilation: Tuple specifying input and output dilation of block.
			mlp_ratio: MLP expansion ratio.
			conv_mlp: Use 1x1 convolutions for MLP and a NCHW compatible norm layer if True.
			conv_bias: Apply bias for all convolution (linear) layers.
			use_grn: Use GlobalResponseNorm in MLP (from ConvNeXt-V2)
			ls_init_value: Layer-scale init values, layer-scale applied if not None.
			act_layer: Activation layer.
			norm_layer: Normalization layer (defaults to LN if not specified).
			drop_path: Stochastic depth probability.
		"""
		super().__init__()
		out_chs = out_chs or in_chs
		dilation = to_ntuple(2)(dilation)
		act_layer = get_act_layer(act_layer)
		if not norm_layer:
			norm_layer = LayerNorm2d if conv_mlp else LayerNorm
		mlp_layer = partial(GlobalResponseNormMlp if use_grn else Mlp, use_conv=conv_mlp)
		self.use_conv_mlp = conv_mlp
		self.conv_dw = create_conv2d(
			in_chs,
			out_chs,
			kernel_size=kernel_size,
			stride=stride,
			dilation=dilation[0],
			depthwise=True if out_chs >= in_chs else False,
			bias=conv_bias,
		)
		self.norm = norm_layer(out_chs)
		self.mlp = mlp_layer(out_chs, int(mlp_ratio * out_chs), act_layer=act_layer)
		self.gamma = nn.Parameter(ls_init_value * torch.ones(out_chs)) if ls_init_value is not None else None
		if in_chs != out_chs or stride != 1 or dilation[0] != dilation[1]:
			self.shortcut = Downsample(in_chs, out_chs, stride=stride, dilation=dilation[0])
		else:
			self.shortcut = nn.Identity()
		self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

	def forward(self, x):
		shortcut = x
		x = self.conv_dw(x)
		if self.use_conv_mlp:
			x = self.norm(x)
			x = self.mlp(x)
		else:
			x = x.permute(0, 2, 3, 1)
			x = self.norm(x)
			x = self.mlp(x)
			x = x.permute(0, 3, 1, 2)
		if self.gamma is not None:
			x = x.mul(self.gamma.reshape(1, -1, 1, 1))

		x = self.drop_path(x) + self.shortcut(shortcut)
		return x


class ConvNeXtStage(nn.Module):

	def __init__(
			self,
			in_chs,
			out_chs,
			kernel_size=7,
			stride=2,
			depth=2,
			dilation=(1, 1),
			drop_path_rates=None,
			ls_init_value=1.0,
			conv_mlp=False,
			conv_bias=True,
			use_grn=False,
			act_layer='gelu',
			norm_layer=None,
			norm_layer_cl=None
	):
		super().__init__()
		self.grad_checkpointing = False

		if in_chs != out_chs or stride > 1 or dilation[0] != dilation[1]:
			ds_ks = 2 if stride > 1 or dilation[0] != dilation[1] else 1
			pad = 'same' if dilation[1] > 1 else 0  # same padding needed if dilation used
			self.downsample = nn.Sequential(
				norm_layer(in_chs),
				create_conv2d(
					in_chs,
					out_chs,
					kernel_size=ds_ks,
					stride=stride,
					dilation=dilation[0],
					padding=pad,
					bias=conv_bias,
				),
			)
			in_chs = out_chs
		else:
			self.downsample = nn.Identity()

		drop_path_rates = drop_path_rates or [0.] * depth
		stage_blocks = []
		for i in range(depth):
			stage_blocks.append(ConvNeXtBlock(
				in_chs=in_chs,
				out_chs=out_chs,
				kernel_size=kernel_size,
				dilation=dilation[1],
				drop_path=drop_path_rates[i],
				ls_init_value=ls_init_value,
				conv_mlp=conv_mlp,
				conv_bias=conv_bias,
				use_grn=use_grn,
				act_layer=act_layer,
				norm_layer=norm_layer if conv_mlp else norm_layer_cl,
			))
			in_chs = out_chs
		self.blocks = nn.Sequential(*stage_blocks)

	def forward(self, x):
		x = self.downsample(x)
		x = self.blocks(x)
		return x


class ConvNeXt(nn.Module):
	r""" ConvNeXt
		A PyTorch impl of : `A ConvNet for the 2020s`  - https://arxiv.org/pdf/2201.03545.pdf
	"""

	def __init__(
			self,
			in_chans: int = 3,
			num_classes: int = 1000,
			global_pool: str = 'avg',
			output_stride: int = 32,
			depths: Tuple[int, ...] = (3, 3, 9, 3),
			dims: Tuple[int, ...] = (96, 192, 384, 768),
			kernel_sizes: Union[int, Tuple[int, ...]] = 7,
			ls_init_value: Optional[float] = 1e-6,
			stem_type: str = 'patch',
			patch_size: int = 4,
			head_init_scale: float = 1.,
			head_norm_first: bool = False,
			head_hidden_size: Optional[int] = None,
			conv_mlp: bool = False,
			conv_bias: bool = True,
			use_grn: bool = False,
			act_layer: Union[str, Callable] = 'gelu',
			norm_layer: Optional[Union[str, Callable]] = None,
			norm_eps: Optional[float] = None,
			drop_rate: float = 0.,
			drop_path_rate: float = 0.,
	):
		"""
		Args:
			in_chans: Number of input image channels.
			num_classes: Number of classes for classification head.
			global_pool: Global pooling type.
			output_stride: Output stride of network, one of (8, 16, 32).
			depths: Number of blocks at each stage.
			dims: Feature dimension at each stage.
			kernel_sizes: Depthwise convolution kernel-sizes for each stage.
			ls_init_value: Init value for Layer Scale, disabled if None.
			stem_type: Type of stem.
			patch_size: Stem patch size for patch stem.
			head_init_scale: Init scaling value for classifier weights and biases.
			head_norm_first: Apply normalization before global pool + head.
			head_hidden_size: Size of MLP hidden layer in head if not None and head_norm_first == False.
			conv_mlp: Use 1x1 conv in MLP, improves speed for small networks w/ chan last.
			conv_bias: Use bias layers w/ all convolutions.
			use_grn: Use Global Response Norm (ConvNeXt-V2) in MLP.
			act_layer: Activation layer type.
			norm_layer: Normalization layer type.
			drop_rate: Head pre-classifier dropout rate.
			drop_path_rate: Stochastic depth drop rate.
		"""
		super().__init__()
		assert output_stride in (8, 16, 32)
		kernel_sizes = to_ntuple(4)(kernel_sizes)
		if norm_layer is None:
			norm_layer = LayerNorm2d
			norm_layer_cl = norm_layer if conv_mlp else LayerNorm
			if norm_eps is not None:
				norm_layer = partial(norm_layer, eps=norm_eps)
				norm_layer_cl = partial(norm_layer_cl, eps=norm_eps)
		else:
			assert conv_mlp,\
				'If a norm_layer is specified, conv MLP must be used so all norm expect rank-4, channels-first input'
			norm_layer_cl = norm_layer
			if norm_eps is not None:
				norm_layer_cl = partial(norm_layer_cl, eps=norm_eps)

		self.num_classes = num_classes
		self.drop_rate = drop_rate
		self.feature_info = []

		assert stem_type in ('patch', 'overlap', 'overlap_tiered')
		if stem_type == 'patch':
			# NOTE: this stem is a minimal form of ViT PatchEmbed, as used in SwinTransformer w/ patch_size = 4
			self.stem = nn.Sequential(
				nn.Conv2d(in_chans, dims[0], kernel_size=patch_size, stride=patch_size, bias=conv_bias),
				norm_layer(dims[0]),
			)
			stem_stride = patch_size
		else:
			mid_chs = make_divisible(dims[0] // 2) if 'tiered' in stem_type else dims[0]
			self.stem = nn.Sequential(
				nn.Conv2d(in_chans, mid_chs, kernel_size=3, stride=2, padding=1, bias=conv_bias),
				nn.Conv2d(mid_chs, dims[0], kernel_size=3, stride=2, padding=1, bias=conv_bias),
				norm_layer(dims[0]),
			)
			stem_stride = 4

		self.stages = nn.Sequential()
		dp_rates = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depths)).split(depths)]
		stages = []
		prev_chs = dims[0]
		curr_stride = stem_stride
		dilation = 1
		# 4 feature resolution stages, each consisting of multiple residual blocks
		for i in range(4):
			stride = 2 if curr_stride == 2 or i > 0 else 1
			if curr_stride >= output_stride and stride > 1:
				dilation *= stride
				stride = 1
			curr_stride *= stride
			first_dilation = 1 if dilation in (1, 2) else 2
			out_chs = dims[i]
			stages.append(ConvNeXtStage(
				prev_chs,
				out_chs,
				kernel_size=kernel_sizes[i],
				stride=stride,
				dilation=(first_dilation, dilation),
				depth=depths[i],
				drop_path_rates=dp_rates[i],
				ls_init_value=ls_init_value,
				conv_mlp=conv_mlp,
				conv_bias=conv_bias,
				use_grn=use_grn,
				act_layer=act_layer,
				norm_layer=norm_layer,
				norm_layer_cl=norm_layer_cl,
			))
			prev_chs = out_chs
			# NOTE feature_info use currently assumes stage 0 == stride 1, rest are stride 2
			self.feature_info += [dict(num_chs=prev_chs, reduction=curr_stride, module=f'stages.{i}')]
		self.stages = nn.Sequential(*stages)
		self.num_features = prev_chs

	@torch.jit.ignore
	def group_matcher(self, coarse=False):
		return dict(
			stem=r'^stem',
			blocks=r'^stages\.(\d+)' if coarse else [
				(r'^stages\.(\d+)\.downsample', (0,)),  # blocks
				(r'^stages\.(\d+)\.blocks\.(\d+)', None),
				(r'^norm_pre', (99999,))
			]
		)

	@torch.jit.ignore
	def set_grad_checkpointing(self, enable=True):
		for s in self.stages:
			s.grad_checkpointing = enable

	@torch.jit.ignore
	def get_classifier(self):
		return self.head.fc

	def forward_features(self, x):
		x = self.stem(x)
		x = self.stages(x)
		return x

def _init_weights(module, name=None, head_init_scale=1.0):
	if isinstance(module, nn.Conv2d):
		trunc_normal_(module.weight, std=.02)
		if module.bias is not None:
			nn.init.zeros_(module.bias)
	elif isinstance(module, nn.Linear):
		trunc_normal_(module.weight, std=.02)
		nn.init.zeros_(module.bias)
		if name and 'head.' in name:
			module.weight.data.mul_(head_init_scale)
			module.bias.data.mul_(head_init_scale)
	
class UpconvSkip(nn.Module) :
	def __init__(self, ch1, ch2, out_ch) -> None:
		super().__init__()
		self.conv = ConvNeXtBlock(
			in_chs=ch1 + ch2,
			out_chs=out_ch,
			kernel_size=7,
			dilation=1,
			drop_path=0,
			ls_init_value=1.0,
			conv_mlp=False,
			conv_bias=True,
			use_grn=False,
			act_layer='gelu',
			norm_layer=LayerNorm,
		)
		self.upconv = nn.ConvTranspose2d(out_ch, out_ch, 2, 2, 0, 0)

	def forward(self, x) :
		x = self.conv(x)
		x = self.upconv(x)
		return x

class DBHead(nn.Module):
	def __init__(self, in_channels, k = 50):
		super().__init__()
		self.k = k
		self.binarize = nn.Sequential(
			nn.Conv2d(in_channels, in_channels // 4, 3, padding=1),
			#nn.BatchNorm2d(in_channels // 4),
			nn.SiLU(inplace=True),
			nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 4, 2, 1),
			#nn.BatchNorm2d(in_channels // 4),
			nn.SiLU(inplace=True),
			nn.ConvTranspose2d(in_channels // 4, 1, 4, 2, 1),
			)
		self.binarize.apply(self.weights_init)

		self.thresh = self._init_thresh(in_channels)
		self.thresh.apply(self.weights_init)

	def forward(self, x):
		shrink_maps = self.binarize(x)
		threshold_maps = self.thresh(x)
		if self.training:
			binary_maps = self.step_function(shrink_maps.sigmoid(), threshold_maps)
			y = torch.cat((shrink_maps, threshold_maps, binary_maps), dim=1)
		else:
			y = torch.cat((shrink_maps, threshold_maps), dim=1)
		return y

	def weights_init(self, m):
		classname = m.__class__.__name__
		if classname.find('Conv') != -1:
			nn.init.kaiming_normal_(m.weight.data)
		elif classname.find('BatchNorm') != -1:
			m.weight.data.fill_(1.)
			m.bias.data.fill_(1e-4)

	def _init_thresh(self, inner_channels, serial=False, smooth=False, bias=False):
		in_channels = inner_channels
		if serial:
			in_channels += 1
		self.thresh = nn.Sequential(
			nn.Conv2d(in_channels, inner_channels // 4, 3, padding=1, bias=bias),
			#nn.GroupNorm(inner_channels // 4),
			nn.SiLU(inplace=True),
			self._init_upsample(inner_channels // 4, inner_channels // 4, smooth=smooth, bias=bias),
			#nn.GroupNorm(inner_channels // 4),
			nn.SiLU(inplace=True),
			self._init_upsample(inner_channels // 4, 1, smooth=smooth, bias=bias),
			nn.Sigmoid())
		return self.thresh

	def _init_upsample(self, in_channels, out_channels, smooth=False, bias=False):
		if smooth:
			inter_out_channels = out_channels
			if out_channels == 1:
				inter_out_channels = in_channels
			module_list = [
				nn.Upsample(scale_factor=2, mode='bilinear'),
				nn.Conv2d(in_channels, inter_out_channels, 3, 1, 1, bias=bias)]
			if out_channels == 1:
				module_list.append(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=1, bias=True))
			return nn.Sequential(module_list)
		else:
			return nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1)

	def step_function(self, x, y):
		return torch.reciprocal(1 + torch.exp(-self.k * (x - y)))

class DBNetConvNext(nn.Module) :
	def __init__(self) :
		super(DBNetConvNext, self).__init__()
		self.backbone = ConvNeXt(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024])

		self.conv_mask = nn.Sequential(
			nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.SiLU(inplace=True),
			nn.Conv2d(64, 32, kernel_size=3, padding=1), nn.SiLU(inplace=True),
			nn.Conv2d(32, 1, kernel_size=1),
			nn.Sigmoid()
		)

		self.down_conv1 = ConvNeXtStage(1024, 1024, depth = 2, norm_layer = LayerNorm2d)
		self.down_conv2 = ConvNeXtStage(1024, 1024, depth = 2, norm_layer = LayerNorm2d)

		self.upconv1 = UpconvSkip(0, 1024, 128)
		self.upconv2 = UpconvSkip(128, 1024, 128)
		self.upconv3 = UpconvSkip(128, 1024, 128)
		self.upconv4 = UpconvSkip(128, 512, 128)
		self.upconv5 = UpconvSkip(128, 256, 128)
		self.upconv6 = UpconvSkip(128, 128, 64)

		self.conv_db = DBHead(128)

	def forward(self, x) :
		# in 3@1536
		x = self.backbone.stem(x) # 128@384
		h4 = self.backbone.stages[0](x) # 128@384
		h8 = self.backbone.stages[1](h4) # 256@192
		h16 = self.backbone.stages[2](h8) # 512@96
		h32 = self.backbone.stages[3](h16) # 1024@48
		h64 = self.down_conv1(h32) # 1024@24
		h128 = self.down_conv2(h64) # 1024@12
		
		up128 = self.upconv1(h128)
		up64 = self.upconv2(torch.cat([up128, h64], dim = 1))
		up32 = self.upconv3(torch.cat([up64, h32], dim = 1))
		up16 = self.upconv4(torch.cat([up32, h16], dim = 1))
		up8 = self.upconv5(torch.cat([up16, h8], dim = 1))
		up4 = self.upconv6(torch.cat([up8, h4], dim = 1))

		return self.conv_db(up8), self.conv_mask(up4)

import os
from .default_utils import imgproc, dbnet_utils, craft_utils
from .common import OfflineDetector
from ..utils import TextBlock, Quadrilateral, det_rearrange_forward

MODEL = None
def det_batch_forward_default(batch: np.ndarray, device: str):
    global MODEL
    if isinstance(batch, list):
        batch = np.array(batch)
    batch = einops.rearrange(batch.astype(np.float32) / 127.5 - 1.0, 'n h w c -> n c h w')
    batch = torch.from_numpy(batch).to(device)
    with torch.no_grad():
        db, mask = MODEL(batch)
        db = db.sigmoid().cpu().numpy()
        mask = mask.cpu().numpy()
    return db, mask


class DBConvNextDetector(OfflineDetector):
    _MODEL_MAPPING = {
        'model': {
            'url': '',
            'hash': '',
            'file': '.',
        }
    }

    def __init__(self, *args, **kwargs):
        os.makedirs(self.model_dir, exist_ok=True)
        if os.path.exists('dbnet_convnext.ckpt'):
            shutil.move('dbnet_convnext.ckpt', self._get_file_path('dbnet_convnext.ckpt'))
        super().__init__(*args, **kwargs)

    async def _load(self, device: str):
        self.model = DBNetConvNext()
        sd = torch.load(self._get_file_path('dbnet_convnext.ckpt'), map_location='cpu')
        self.model.load_state_dict(sd['model'] if 'model' in sd else sd)
        self.model.eval()
        self.device = device
        if device == 'cuda' or device == 'mps':
            self.model = self.model.to(self.device)
        global MODEL
        MODEL = self.model

    async def _unload(self):
        del self.model

    async def _infer(self, image: np.ndarray, detect_size: int, text_threshold: float, box_threshold: float,
                     unclip_ratio: float, verbose: bool = False):

        # TODO: Move det_rearrange_forward to common.py and refactor
        db, mask = det_rearrange_forward(image, det_batch_forward_default, detect_size, 4, device=self.device, verbose=verbose)

        if db is None:
            # rearrangement is not required, fallback to default forward
            img_resized, target_ratio, _, pad_w, pad_h = imgproc.resize_aspect_ratio(cv2.bilateralFilter(image, 17, 80, 80), detect_size, cv2.INTER_LINEAR, mag_ratio = 1)
            img_resized_h, img_resized_w = img_resized.shape[:2]
            ratio_h = ratio_w = 1 / target_ratio
            db, mask = det_batch_forward_default([img_resized], self.device)
        else:
            img_resized_h, img_resized_w = image.shape[:2]
            ratio_w = ratio_h = 1
            pad_h = pad_w = 0
        self.logger.info(f'Detection resolution: {img_resized_w}x{img_resized_h}')

        mask = mask[0, 0, :, :]
        det = dbnet_utils.SegDetectorRepresenter(text_threshold, box_threshold, unclip_ratio=unclip_ratio)
        # boxes, scores = det({'shape': [(img_resized.shape[0], img_resized.shape[1])]}, db)
        boxes, scores = det({'shape':[(img_resized_h, img_resized_w)]}, db)
        boxes, scores = boxes[0], scores[0]
        if boxes.size == 0:
            polys = []
        else:
            idx = boxes.reshape(boxes.shape[0], -1).sum(axis=1) > 0
            polys, _ = boxes[idx], scores[idx]
            polys = polys.astype(np.float64)
            polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h, ratio_net=1)
            polys = polys.astype(np.int16)

        textlines = [Quadrilateral(pts.astype(int), '', score) for pts, score in zip(polys, scores)]
        textlines = list(filter(lambda q: q.area > 16, textlines))
        mask_resized = cv2.resize(mask, (mask.shape[1] * 2, mask.shape[0] * 2), interpolation=cv2.INTER_LINEAR)
        if pad_h > 0:
            mask_resized = mask_resized[:-pad_h, :]
        elif pad_w > 0:
            mask_resized = mask_resized[:, :-pad_w]
        raw_mask = np.clip(mask_resized * 255, 0, 255).astype(np.uint8)

        # if verbose:
        #     img_bbox_raw = np.copy(image)
        #     for txtln in textlines:
        #         cv2.polylines(img_bbox_raw, [txtln.pts], True, color=(255, 0, 0), thickness=2)
        #     cv2.imwrite(f'result/bboxes_unfiltered.png', cv2.cvtColor(img_bbox_raw, cv2.COLOR_RGB2BGR))

        return textlines, raw_mask, None


if __name__ == '__main__' :
	net = DBNetConvNext().cuda()
	img = torch.randn(2, 3, 1536, 1536).cuda()
	ret1, ret2 = net.forward(img)
	print(ret1.shape)
	print(ret2.shape)
