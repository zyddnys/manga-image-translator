
from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

def relu_nf(x) :
	return F.relu(x) * 1.7139588594436646

def gelu_nf(x) :
	return F.gelu(x) * 1.7015043497085571

def silu_nf(x) :
	return F.silu(x) * 1.7881293296813965

class LambdaLayer(nn.Module) :
	def __init__(self, f):
		super(LambdaLayer, self).__init__()
		self.f = f

	def forward(self, x) :
		return self.f(x)

class ScaledWSConv2d(nn.Conv2d):
	"""2D Conv layer with Scaled Weight Standardization."""
	def __init__(self, in_channels, out_channels, kernel_size,
		stride=1, padding=0,
		dilation=1, groups=1, bias=True, gain=True,
		eps=1e-4):
		nn.Conv2d.__init__(self, in_channels, out_channels,
			kernel_size, stride,
			padding, dilation,
			groups, bias)
		#nn.init.kaiming_normal_(self.weight)
		if gain:
			self.gain = nn.Parameter(torch.ones(self.out_channels, 1, 1, 1))
		else:
			self.gain = None
		# Epsilon, a small constant to avoid dividing by zero.
		self.eps = eps
	def get_weight(self):
		# Get Scaled WS weight OIHW;
		fan_in = np.prod(self.weight.shape[1:])
		var, mean = torch.var_mean(self.weight, dim=(1, 2, 3), keepdims=True)
		scale = torch.rsqrt(torch.max(
			var * fan_in, torch.tensor(self.eps).to(var.device))) * self.gain.view_as(var).to(var.device)
		shift = mean * scale
		return self.weight * scale - shift
		
	def forward(self, x):
		return F.conv2d(x, self.get_weight(), self.bias,
			self.stride, self.padding,
			self.dilation, self.groups)

class ScaledWSTransposeConv2d(nn.ConvTranspose2d):
	"""2D Transpose Conv layer with Scaled Weight Standardization."""
	def __init__(self, in_channels: int,
		out_channels: int,
		kernel_size,
		stride = 1,
		padding = 0,
		output_padding = 0,
		groups: int = 1,
		bias: bool = True,
		dilation: int = 1,
		gain=True,
		eps=1e-4):
		nn.ConvTranspose2d.__init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, groups, bias, dilation, 'zeros')
		#nn.init.kaiming_normal_(self.weight)
		if gain:
			self.gain = nn.Parameter(torch.ones(self.in_channels, 1, 1, 1))
		else:
			self.gain = None
		# Epsilon, a small constant to avoid dividing by zero.
		self.eps = eps
	def get_weight(self):
		# Get Scaled WS weight OIHW;
		fan_in = np.prod(self.weight.shape[1:])
		var, mean = torch.var_mean(self.weight, dim=(1, 2, 3), keepdims=True)
		scale = torch.rsqrt(torch.max(
			var * fan_in, torch.tensor(self.eps).to(var.device))) * self.gain.view_as(var).to(var.device)
		shift = mean * scale
		return self.weight * scale - shift
		
	def forward(self, x, output_size: Optional[List[int]] = None):
		output_padding = self._output_padding(
			input, output_size, self.stride, self.padding, self.kernel_size, self.dilation)
		return F.conv_transpose2d(x, self.get_weight(), self.bias, self.stride, self.padding,
			output_padding, self.groups, self.dilation)

class GatedWSConvPadded(nn.Module) :
	def __init__(self, in_ch, out_ch, ks, stride = 1, dilation = 1) :
		super(GatedWSConvPadded, self).__init__()
		self.in_ch = in_ch
		self.out_ch = out_ch
		self.padding = nn.ReflectionPad2d((ks - 1) // 2)
		self.conv = ScaledWSConv2d(in_ch, out_ch, kernel_size = ks, stride = stride)
		self.conv_gate = ScaledWSConv2d(in_ch, out_ch, kernel_size = ks, stride = stride)

	def forward(self, x) :
		x = self.padding(x)
		signal = self.conv(x)
		gate = torch.sigmoid(self.conv_gate(x))
		return signal * gate * 1.8

class GatedWSTransposeConvPadded(nn.Module) :
	def __init__(self, in_ch, out_ch, ks, stride = 1) :
		super(GatedWSTransposeConvPadded, self).__init__()
		self.in_ch = in_ch
		self.out_ch = out_ch
		self.conv = ScaledWSTransposeConv2d(in_ch, out_ch, kernel_size = ks, stride = stride, padding = (ks - 1) // 2)
		self.conv_gate = ScaledWSTransposeConv2d(in_ch, out_ch, kernel_size = ks, stride = stride, padding = (ks - 1) // 2)

	def forward(self, x) :
		signal = self.conv(x)
		gate = torch.sigmoid(self.conv_gate(x))
		return signal * gate * 1.8

class ResBlock(nn.Module) :
	def __init__(self, ch, alpha = 0.2, beta = 1.0, dilation = 1) :
		super(ResBlock, self).__init__()
		self.alpha = alpha
		self.beta = beta
		self.c1 = GatedWSConvPadded(ch, ch, 3, dilation = dilation)
		self.c2 = GatedWSConvPadded(ch, ch, 3, dilation = dilation)

	def forward(self, x) :
		skip = x
		x = self.c1(relu_nf(x / self.beta))
		x = self.c2(relu_nf(x))
		x = x * self.alpha
		return x + skip

# from https://github.com/SayedNadim/Global-and-Local-Attention-Based-Free-Form-Image-Inpainting
# Contextual attention implementation is borrowed from IJCAI 2019 : "MUSICAL: Multi-Scale Image Contextual Attention Learning for Inpainting".
# Original implementation causes bad results for Pytorch 1.2+.
class GlobalLocalAttention(nn.Module):
	def __init__(self, in_dim, patch_size=3, propagate_size=3, stride=1):
		super(GlobalLocalAttention, self).__init__()
		self.patch_size = patch_size
		self.propagate_size = propagate_size
		self.stride = stride
		self.prop_kernels = None
		self.in_dim = in_dim
		self.feature_attention = GlobalAttention(in_dim)
		self.patch_attention = GlobalAttentionPatch(in_dim)

	def forward(self, foreground, mask, background="same"):
		###assume the masked area has value 1
		bz, nc, w, h = foreground.size()
		if background == "same":
			background = foreground.clone()
		mask = F.interpolate(mask, size=(h, w), mode='nearest')
		background = background * (1 - mask)
		foreground = self.feature_attention(foreground, background, mask)
		background = F.pad(background,
						   [self.patch_size // 2, self.patch_size // 2, self.patch_size // 2, self.patch_size // 2])
		conv_kernels_all = background.unfold(2, self.patch_size, self.stride).unfold(3, self.patch_size,
																					 self.stride).contiguous().view(bz,
																													nc,
																													-1,
																													self.patch_size,
																													self.patch_size)

		mask_resized = mask.repeat(1, self.in_dim, 1, 1)
		mask_resized = F.pad(mask_resized,
							 [self.patch_size // 2, self.patch_size // 2, self.patch_size // 2, self.patch_size // 2])
		mask_kernels_all = mask_resized.unfold(2, self.patch_size, self.stride).unfold(3, self.patch_size,
																					   self.stride).contiguous().view(
			bz,
			nc,
			-1,
			self.patch_size,
			self.patch_size)
		conv_kernels_all = conv_kernels_all.transpose(2, 1)
		mask_kernels_all = mask_kernels_all.transpose(2, 1)
		output_tensor = []
		for i in range(bz):
			feature_map = foreground[i:i + 1]

			# form convolutional kernels
			conv_kernels = conv_kernels_all[i] + 0.0000001
			mask_kernels = mask_kernels_all[i]
			conv_kernels = self.patch_attention(conv_kernels, conv_kernels, mask_kernels)
			norm_factor = torch.sum(conv_kernels ** 2, [1, 2, 3], keepdim=True) ** 0.5
			conv_kernels = conv_kernels / norm_factor

			conv_result = F.conv2d(feature_map, conv_kernels, padding=self.patch_size // 2)
			if self.propagate_size != 1:
				if self.prop_kernels is None:
					self.prop_kernels = torch.ones([conv_result.size(1), 1, self.propagate_size, self.propagate_size],device = mask.device)
					self.prop_kernels.requires_grad = False
					self.prop_kernels = self.prop_kernels
				conv_result = F.conv2d(conv_result, self.prop_kernels, stride=1, padding=1, groups=conv_result.size(1))
			mm = (torch.mean(mask_kernels_all[i], dim=[1,2,3], keepdim=True)==0.0).to(torch.float32)
			mm = mm.permute(1,0,2,3)
			conv_result = conv_result * mm
			attention_scores = F.softmax(conv_result, dim=1)
			attention_scores = attention_scores * mm

			##propagate the scores
			recovered_foreground = F.conv_transpose2d(attention_scores, conv_kernels, stride=1,
													  padding=self.patch_size // 2)
			output_tensor.append(recovered_foreground)
		return torch.cat(output_tensor, dim=0)

# from https://github.com/SayedNadim/Global-and-Local-Attention-Based-Free-Form-Image-Inpainting
class GlobalAttention(nn.Module):
	""" Self attention Layer"""

	def __init__(self, in_dim):
		super(GlobalAttention, self).__init__()
		self.chanel_in = in_dim

		self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
		self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
		self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
		self.softmax = nn.Softmax(dim=-1)  #
		self.rate = 1
		self.gamma = nn.parameter.Parameter(torch.tensor([1.0], requires_grad=True), requires_grad=True)

	def forward(self, a, b, c):
		m_batchsize, C, width, height = a.size()  # B, C, H, W
		down_rate = int(c.size(2)//width)
		c = F.interpolate(c, scale_factor=1./down_rate*self.rate, mode='nearest')
		proj_query = self.query_conv(a).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B, C, N -> B N C
		proj_key = self.key_conv(b).view(m_batchsize, -1, width * height)  # B, C, N
		feature_similarity = torch.bmm(proj_query, proj_key)  # B, N, N

		mask = c.view(m_batchsize, -1, width * height)  # B, C, N
		mask = mask.repeat(1, height * width, 1).permute(0, 2, 1)  # B, 1, H, W -> B, C, H, W // B

		feature_pruning = feature_similarity * mask
		attention = self.softmax(feature_pruning)  # B, N, C
		feature_pruning = torch.bmm(self.value_conv(a).view(m_batchsize, -1, width * height),
									attention.permute(0, 2, 1))  # -. B, C, N
		out = feature_pruning.view(m_batchsize, C, width, height)  # B, C, H, W
		out = a * c + self.gamma *  (1.0 - c) * out
		return out


class GlobalAttentionPatch(nn.Module):
	""" Self attention Layer"""

	def __init__(self, in_dim):
		super(GlobalAttentionPatch, self).__init__()
		self.chanel_in = in_dim

		self.query_channel = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
		self.key_channel = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
		self.value_channel = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

		self.softmax_channel = nn.Softmax(dim=-1)
		self.gamma = nn.parameter.Parameter(torch.tensor([1.0], requires_grad=True), requires_grad=True)

	def forward(self, x, y, m):
		'''
		Something
		'''
		feature_size = list(x.size())
		# Channel attention
		query_channel = self.query_channel(x).view(feature_size[0], -1, feature_size[2] * feature_size[3])
		key_channel = self.key_channel(y).view(feature_size[0], -1, feature_size[2] * feature_size[3]).permute(0,
																											   2,
																											   1)
		channel_correlation = torch.bmm(query_channel, key_channel)
		m_r = m.view(feature_size[0], -1, feature_size[2]*feature_size[3])
		channel_correlation = torch.bmm(channel_correlation, m_r)
		energy_channel = self.softmax_channel(channel_correlation)
		value_channel = self.value_channel(x).view(feature_size[0], -1, feature_size[2] * feature_size[3])
		attented_channel = (energy_channel * value_channel).view(feature_size[0], feature_size[1],
																		 feature_size[2],
																		 feature_size[3])
		out = x * m + self.gamma * (1.0 - m) * attented_channel
		return out


class CoarseGenerator(nn.Module) :
	def __init__(self, in_ch = 4, out_ch = 3, ch = 32, alpha = 0.2) :
		super(CoarseGenerator, self).__init__()

		self.head = nn.Sequential(
			GatedWSConvPadded(in_ch, ch, 3, stride = 1),
			LambdaLayer(relu_nf),
			GatedWSConvPadded(ch, ch * 2, 4, stride = 2),
			LambdaLayer(relu_nf),
			GatedWSConvPadded(ch * 2, ch * 4, 4, stride = 2),
		)

		self.beta = 1.0
		self.alpha = alpha
		self.body_conv = []
		self.body_conv.append(ResBlock(ch * 4, self.alpha, self.beta))
		self.beta = (self.beta ** 2 + self.alpha ** 2) ** 0.5
		self.body_conv.append(ResBlock(ch * 4, self.alpha, self.beta))
		self.beta = (self.beta ** 2 + self.alpha ** 2) ** 0.5
		self.body_conv.append(ResBlock(ch * 4, self.alpha, self.beta, 2))
		self.beta = (self.beta ** 2 + self.alpha ** 2) ** 0.5
		self.body_conv.append(ResBlock(ch * 4, self.alpha, self.beta, 4))
		self.beta = (self.beta ** 2 + self.alpha ** 2) ** 0.5
		self.body_conv.append(ResBlock(ch * 4, self.alpha, self.beta, 8))
		self.beta = (self.beta ** 2 + self.alpha ** 2) ** 0.5
		self.body_conv.append(ResBlock(ch * 4, self.alpha, self.beta, 16))
		self.beta = (self.beta ** 2 + self.alpha ** 2) ** 0.5
		self.body_conv = nn.Sequential(*self.body_conv)

		self.tail = nn.Sequential(
			LambdaLayer(relu_nf),
			GatedWSConvPadded(ch * 8, ch * 4, 3, 1),
			LambdaLayer(relu_nf),
			GatedWSTransposeConvPadded(ch * 4, ch * 2, 4, 2),
			LambdaLayer(relu_nf),
			GatedWSTransposeConvPadded(ch * 2, ch, 4, 2),
			LambdaLayer(relu_nf),
			GatedWSConvPadded(ch, out_ch, 3, stride = 1),
		)

		self.beta = 1.0

		self.body_attn_1 = ResBlock(ch * 4, self.alpha, self.beta)
		self.beta = (self.beta ** 2 + self.alpha ** 2) ** 0.5
		self.body_attn_2 = ResBlock(ch * 4, self.alpha, self.beta)
		self.beta = (self.beta ** 2 + self.alpha ** 2) ** 0.5
		self.body_attn_3 = ResBlock(ch * 4, self.alpha, self.beta)
		self.beta = (self.beta ** 2 + self.alpha ** 2) ** 0.5
		self.body_attn_4 = ResBlock(ch * 4, self.alpha, self.beta)
		self.beta = (self.beta ** 2 + self.alpha ** 2) ** 0.5
		self.body_attn_attn = GlobalAttention(in_dim = ch * 4)
		self.body_attn_5 = ResBlock(ch * 4, self.alpha, self.beta)
		self.beta = (self.beta ** 2 + self.alpha ** 2) ** 0.5
		self.body_attn_6 = ResBlock(ch * 4, self.alpha, self.beta)
		self.beta = (self.beta ** 2 + self.alpha ** 2) ** 0.5

	def forward(self, img, mask) :
		x = torch.cat([mask, img], dim = 1)
		x = self.head(x)
		attn = self.body_attn_1(x)
		attn = self.body_attn_2(attn)
		attn = self.body_attn_3(attn)
		attn = self.body_attn_4(attn)
		attn = self.body_attn_attn(attn, attn, mask)
		attn = self.body_attn_5(attn)
		attn = self.body_attn_6(attn)
		conv = self.body_conv(x)
		x = self.tail(torch.cat([conv, attn], dim = 1))
		return torch.clip(x, -1, 1)

class RefineGenerator(nn.Module) :
	def __init__(self, in_ch = 5, out_ch = 3, ch = 64, alpha = 0.2) :
		super(RefineGenerator, self).__init__()

		self.head = nn.Sequential(
			GatedWSConvPadded(in_ch, ch, 3, stride = 1),
			LambdaLayer(relu_nf),
			GatedWSConvPadded(ch, ch * 2, 4, stride = 2),
			LambdaLayer(relu_nf),
			GatedWSConvPadded(ch * 2, ch * 4, 4, stride = 2),
		)

		self.beta = 1.0
		self.alpha = alpha
		self.body_conv = []
		self.body_conv.append(ResBlock(ch * 4, self.alpha, self.beta))
		self.beta = (self.beta ** 2 + self.alpha ** 2) ** 0.5
		self.body_conv.append(ResBlock(ch * 4, self.alpha, self.beta))
		self.beta = (self.beta ** 2 + self.alpha ** 2) ** 0.5
		self.body_conv.append(ResBlock(ch * 4, self.alpha, self.beta, 2))
		self.beta = (self.beta ** 2 + self.alpha ** 2) ** 0.5
		self.body_conv.append(ResBlock(ch * 4, self.alpha, self.beta, 4))
		self.beta = (self.beta ** 2 + self.alpha ** 2) ** 0.5
		self.body_conv.append(ResBlock(ch * 4, self.alpha, self.beta, 8))
		self.beta = (self.beta ** 2 + self.alpha ** 2) ** 0.5
		self.body_conv.append(ResBlock(ch * 4, self.alpha, self.beta, 16))
		self.beta = (self.beta ** 2 + self.alpha ** 2) ** 0.5
		self.body_conv.append(ResBlock(ch * 4, self.alpha, self.beta))
		self.beta = (self.beta ** 2 + self.alpha ** 2) ** 0.5
		self.body_conv = nn.Sequential(*self.body_conv)

		self.tail = nn.Sequential(
			LambdaLayer(relu_nf),
			GatedWSConvPadded(ch * 8, ch * 4, 3, 1),
			LambdaLayer(relu_nf),
			GatedWSTransposeConvPadded(ch * 4, ch * 2, 4, 2),
			LambdaLayer(relu_nf),
			GatedWSTransposeConvPadded(ch * 2, ch, 4, 2),
			LambdaLayer(relu_nf),
			GatedWSConvPadded(ch, out_ch, 3, stride = 1),
		)

		self.beta = 1.0

		self.body_attn_1 = ResBlock(ch * 4, self.alpha, self.beta)
		self.beta = (self.beta ** 2 + self.alpha ** 2) ** 0.5
		self.body_attn_2 = ResBlock(ch * 4, self.alpha, self.beta)
		self.beta = (self.beta ** 2 + self.alpha ** 2) ** 0.5
		self.body_attn_3 = ResBlock(ch * 4, self.alpha, self.beta)
		self.beta = (self.beta ** 2 + self.alpha ** 2) ** 0.5
		self.body_attn_4 = ResBlock(ch * 4, self.alpha, self.beta)
		self.beta = (self.beta ** 2 + self.alpha ** 2) ** 0.5
		self.body_attn_attn = GlobalLocalAttention(in_dim = ch * 4)
		self.body_attn_5 = ResBlock(ch * 4, self.alpha, self.beta)
		self.beta = (self.beta ** 2 + self.alpha ** 2) ** 0.5
		self.body_attn_6 = ResBlock(ch * 4, self.alpha, self.beta)
		self.beta = (self.beta ** 2 + self.alpha ** 2) ** 0.5
		self.body_attn_7 = ResBlock(ch * 4, self.alpha, self.beta)
		self.beta = (self.beta ** 2 + self.alpha ** 2) ** 0.5

	def forward(self, img, img_coarse, mask) :
		x = img * mask + img_coarse * (1. - mask)
		x = torch.cat([mask, x], dim = 1)
		x = self.head(x)
		attn = self.body_attn_1(x)
		attn = self.body_attn_2(attn)
		attn = self.body_attn_3(attn)
		attn = self.body_attn_4(attn)
		attn = self.body_attn_attn(attn, mask)
		attn = self.body_attn_5(attn)
		attn = self.body_attn_6(attn)
		attn = self.body_attn_7(attn)
		conv = self.body_conv(x)
		x = self.tail(torch.cat([conv, attn], dim = 1))
		return torch.clip(x, -1, 1)

class InpaintingVanilla(nn.Module):
	def __init__(self):
		super(InpaintingVanilla, self).__init__()
		self.coarse_generator = CoarseGenerator(4, 3, 32)
		self.fine_generator = RefineGenerator(4, 3, 64)

	def forward(self, x, mask):
		x_stage1 = self.coarse_generator(x, mask)
		x_stage2 = self.fine_generator(x, x_stage1, mask)
		return x_stage1, x_stage2
