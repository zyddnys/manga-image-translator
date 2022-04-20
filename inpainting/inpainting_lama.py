# Fast Fourier Convolution NeurIPS 2020
# original implementation https://github.com/pkumivision/FFC/blob/main/model_zoo/ffc.py
# paper https://proceedings.neurips.cc/paper/2020/file/2fd5d41ec6cfab47e32164d5624269b1-Paper.pdf

from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import abc
import random
from kornia.geometry.transform import rotate


class DepthWiseSeperableConv(nn.Module):
	def __init__(self, in_dim, out_dim, *args, **kwargs):
		super().__init__()
		if 'groups' in kwargs:
			# ignoring groups for Depthwise Sep Conv
			del kwargs['groups']
		
		self.depthwise = nn.Conv2d(in_dim, in_dim, *args, groups=in_dim, **kwargs)
		self.pointwise = nn.Conv2d(in_dim, out_dim, kernel_size=1)
		
	def forward(self, x):
		out = self.depthwise(x)
		out = self.pointwise(out)
		return out

class MultidilatedConv(nn.Module):
	def __init__(self, in_dim, out_dim, kernel_size, dilation_num=3, comb_mode='sum', equal_dim=True,
				 shared_weights=False, padding=1, min_dilation=1, shuffle_in_channels=False, use_depthwise=False, **kwargs):
		super().__init__()
		convs = []
		self.equal_dim = equal_dim
		assert comb_mode in ('cat_out', 'sum', 'cat_in', 'cat_both'), comb_mode
		if comb_mode in ('cat_out', 'cat_both'):
			self.cat_out = True
			if equal_dim:
				assert out_dim % dilation_num == 0
				out_dims = [out_dim // dilation_num] * dilation_num
				self.index = sum([[i + j * (out_dims[0]) for j in range(dilation_num)] for i in range(out_dims[0])], [])
			else:
				out_dims = [out_dim // 2 ** (i + 1) for i in range(dilation_num - 1)]
				out_dims.append(out_dim - sum(out_dims))
				index = []
				starts = [0] + out_dims[:-1]
				lengths = [out_dims[i] // out_dims[-1] for i in range(dilation_num)]
				for i in range(out_dims[-1]):
					for j in range(dilation_num):
						index += list(range(starts[j], starts[j] + lengths[j]))
						starts[j] += lengths[j]
				self.index = index
				assert(len(index) == out_dim)
			self.out_dims = out_dims
		else:
			self.cat_out = False
			self.out_dims = [out_dim] * dilation_num

		if comb_mode in ('cat_in', 'cat_both'):
			if equal_dim:
				assert in_dim % dilation_num == 0
				in_dims = [in_dim // dilation_num] * dilation_num
			else:
				in_dims = [in_dim // 2 ** (i + 1) for i in range(dilation_num - 1)]
				in_dims.append(in_dim - sum(in_dims))
			self.in_dims = in_dims
			self.cat_in = True
		else:
			self.cat_in = False
			self.in_dims = [in_dim] * dilation_num

		conv_type = DepthWiseSeperableConv if use_depthwise else nn.Conv2d
		dilation = min_dilation
		for i in range(dilation_num):
			if isinstance(padding, int):
				cur_padding = padding * dilation
			else:
				cur_padding = padding[i]
			convs.append(conv_type(
				self.in_dims[i], self.out_dims[i], kernel_size, padding=cur_padding, dilation=dilation, **kwargs
			))
			if i > 0 and shared_weights:
				convs[-1].weight = convs[0].weight
				convs[-1].bias = convs[0].bias
			dilation *= 2
		self.convs = nn.ModuleList(convs)

		self.shuffle_in_channels = shuffle_in_channels
		if self.shuffle_in_channels:
			# shuffle list as shuffling of tensors is nondeterministic
			in_channels_permute = list(range(in_dim))
			random.shuffle(in_channels_permute)
			# save as buffer so it is saved and loaded with checkpoint
			self.register_buffer('in_channels_permute', torch.tensor(in_channels_permute))

	def forward(self, x):
		if self.shuffle_in_channels:
			x = x[:, self.in_channels_permute]

		outs = []
		if self.cat_in:
			if self.equal_dim:
				x = x.chunk(len(self.convs), dim=1)
			else:
				new_x = []
				start = 0
				for dim in self.in_dims:
					new_x.append(x[:, start:start+dim])
					start += dim
				x = new_x
		for i, conv in enumerate(self.convs):
			if self.cat_in:
				input = x[i]
			else:
				input = x
			outs.append(conv(input))
		if self.cat_out:
			out = torch.cat(outs, dim=1)[:, self.index]
		else:
			out = sum(outs)
		return out

class BaseDiscriminator(nn.Module):
	@abc.abstractmethod
	def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
		"""
		Predict scores and get intermediate activations. Useful for feature matching loss
		:return tuple (scores, list of intermediate activations)
		"""
		raise NotImplemented()


def get_conv_block_ctor(kind='default'):
	if not isinstance(kind, str):
		return kind
	if kind == 'default':
		return nn.Conv2d
	if kind == 'depthwise':
		return DepthWiseSeperableConv   
	if kind == 'multidilated':
		return MultidilatedConv
	raise ValueError(f'Unknown convolutional block kind {kind}')


def get_norm_layer(kind='bn'):
	if not isinstance(kind, str):
		return kind
	if kind == 'bn':
		return nn.BatchNorm2d
	if kind == 'in':
		return nn.InstanceNorm2d
	raise ValueError(f'Unknown norm block kind {kind}')


def get_activation(kind='tanh'):
	if kind == 'tanh':
		return nn.Tanh()
	if kind == 'sigmoid':
		return nn.Sigmoid()
	if kind is False:
		return nn.Identity()
	raise ValueError(f'Unknown activation kind {kind}')

class LearnableSpatialTransformWrapper(nn.Module):
	def __init__(self, impl, pad_coef=0.5, angle_init_range=80, train_angle=True):
		super().__init__()
		self.impl = impl
		self.angle = torch.rand(1) * angle_init_range
		if train_angle:
			self.angle = nn.Parameter(self.angle, requires_grad=True)
		self.pad_coef = pad_coef

	def forward(self, x):
		if torch.is_tensor(x):
			return self.inverse_transform(self.impl(self.transform(x)), x)
		elif isinstance(x, tuple):
			x_trans = tuple(self.transform(elem) for elem in x)
			y_trans = self.impl(x_trans)
			return tuple(self.inverse_transform(elem, orig_x) for elem, orig_x in zip(y_trans, x))
		else:
			raise ValueError(f'Unexpected input type {type(x)}')

	def transform(self, x):
		height, width = x.shape[2:]
		pad_h, pad_w = int(height * self.pad_coef), int(width * self.pad_coef)
		x_padded = F.pad(x, [pad_w, pad_w, pad_h, pad_h], mode='reflect')
		x_padded_rotated = rotate(x_padded, angle=self.angle.to(x_padded))
		return x_padded_rotated

	def inverse_transform(self, y_padded_rotated, orig_x):
		height, width = orig_x.shape[2:]
		pad_h, pad_w = int(height * self.pad_coef), int(width * self.pad_coef)

		y_padded = rotate(y_padded_rotated, angle=-self.angle.to(y_padded_rotated))
		y_height, y_width = y_padded.shape[2:]
		y = y_padded[:, :, pad_h : y_height - pad_h, pad_w : y_width - pad_w]
		return y

class FFCSE_block(nn.Module):

	def __init__(self, channels, ratio_g):
		super(FFCSE_block, self).__init__()
		in_cg = int(channels * ratio_g)
		in_cl = channels - in_cg
		r = 16

		self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
		self.conv1 = nn.Conv2d(channels, channels // r,
							   kernel_size=1, bias=True)
		self.relu1 = nn.ReLU(inplace=True)
		self.conv_a2l = None if in_cl == 0 else nn.Conv2d(
			channels // r, in_cl, kernel_size=1, bias=True)
		self.conv_a2g = None if in_cg == 0 else nn.Conv2d(
			channels // r, in_cg, kernel_size=1, bias=True)
		self.sigmoid = nn.Sigmoid()

	def forward(self, x):
		x = x if type(x) is tuple else (x, 0)
		id_l, id_g = x

		x = id_l if type(id_g) is int else torch.cat([id_l, id_g], dim=1)
		x = self.avgpool(x)
		x = self.relu1(self.conv1(x))

		x_l = 0 if self.conv_a2l is None else id_l * \
			self.sigmoid(self.conv_a2l(x))
		x_g = 0 if self.conv_a2g is None else id_g * \
			self.sigmoid(self.conv_a2g(x))
		return x_l, x_g

class SELayer(nn.Module):
	def __init__(self, channel, reduction=16):
		super(SELayer, self).__init__()
		self.avg_pool = nn.AdaptiveAvgPool2d(1)
		self.fc = nn.Sequential(
			nn.Linear(channel, channel // reduction, bias=False),
			nn.ReLU(inplace=True),
			nn.Linear(channel // reduction, channel, bias=False),
			nn.Sigmoid()
		)

	def forward(self, x):
		b, c, _, _ = x.size()
		y = self.avg_pool(x).view(b, c)
		y = self.fc(y).view(b, c, 1, 1)
		res = x * y.expand_as(x)
		return res

class FourierUnit(nn.Module):

	def __init__(self, in_channels, out_channels, groups=1, spatial_scale_factor=None, spatial_scale_mode='bilinear',
				 spectral_pos_encoding=False, use_se=False, se_kwargs=None, ffc3d=False, fft_norm='ortho'):
		# bn_layer not used
		super(FourierUnit, self).__init__()
		self.groups = groups

		self.conv_layer = torch.nn.Conv2d(in_channels=in_channels * 2 + (2 if spectral_pos_encoding else 0),
										  out_channels=out_channels * 2,
										  kernel_size=1, stride=1, padding=0, groups=self.groups, bias=False)
		self.bn = torch.nn.BatchNorm2d(out_channels * 2)
		self.relu = torch.nn.ReLU(inplace=True)

		# squeeze and excitation block
		self.use_se = use_se
		if use_se:
			if se_kwargs is None:
				se_kwargs = {}
			self.se = SELayer(self.conv_layer.in_channels, **se_kwargs)

		self.spatial_scale_factor = spatial_scale_factor
		self.spatial_scale_mode = spatial_scale_mode
		self.spectral_pos_encoding = spectral_pos_encoding
		self.ffc3d = ffc3d
		self.fft_norm = fft_norm

	def forward(self, x):
		batch = x.shape[0]

		if self.spatial_scale_factor is not None:
			orig_size = x.shape[-2:]
			x = F.interpolate(x, scale_factor=self.spatial_scale_factor, mode=self.spatial_scale_mode, align_corners=False)

		r_size = x.size()
		# (batch, c, h, w/2+1, 2)
		fft_dim = (-3, -2, -1) if self.ffc3d else (-2, -1)
		ffted = torch.fft.rfftn(x, dim=fft_dim, norm=self.fft_norm)
		ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
		ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()  # (batch, c, 2, h, w/2+1)
		ffted = ffted.view((batch, -1,) + ffted.size()[3:])

		if self.spectral_pos_encoding:
			height, width = ffted.shape[-2:]
			coords_vert = torch.linspace(0, 1, height)[None, None, :, None].expand(batch, 1, height, width).to(ffted)
			coords_hor = torch.linspace(0, 1, width)[None, None, None, :].expand(batch, 1, height, width).to(ffted)
			ffted = torch.cat((coords_vert, coords_hor, ffted), dim=1)

		if self.use_se:
			ffted = self.se(ffted)

		ffted = self.conv_layer(ffted)  # (batch, c*2, h, w/2+1)
		ffted = self.relu(self.bn(ffted))

		ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(
			0, 1, 3, 4, 2).contiguous()  # (batch,c, t, h, w/2+1, 2)
		ffted = torch.complex(ffted[..., 0], ffted[..., 1])

		ifft_shape_slice = x.shape[-3:] if self.ffc3d else x.shape[-2:]
		output = torch.fft.irfftn(ffted, s=ifft_shape_slice, dim=fft_dim, norm=self.fft_norm)

		if self.spatial_scale_factor is not None:
			output = F.interpolate(output, size=orig_size, mode=self.spatial_scale_mode, align_corners=False)

		return output


class SpectralTransform(nn.Module):

	def __init__(self, in_channels, out_channels, stride=1, groups=1, enable_lfu=True, **fu_kwargs):
		# bn_layer not used
		super(SpectralTransform, self).__init__()
		self.enable_lfu = enable_lfu
		if stride == 2:
			self.downsample = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
		else:
			self.downsample = nn.Identity()

		self.stride = stride
		self.conv1 = nn.Sequential(
			nn.Conv2d(in_channels, out_channels //
					  2, kernel_size=1, groups=groups, bias=False),
			nn.BatchNorm2d(out_channels // 2),
			nn.ReLU(inplace=True)
		)
		self.fu = FourierUnit(
			out_channels // 2, out_channels // 2, groups, **fu_kwargs)
		if self.enable_lfu:
			self.lfu = FourierUnit(
				out_channels // 2, out_channels // 2, groups)
		self.conv2 = torch.nn.Conv2d(
			out_channels // 2, out_channels, kernel_size=1, groups=groups, bias=False)

	def forward(self, x):

		x = self.downsample(x)
		x = self.conv1(x)
		output = self.fu(x)

		if self.enable_lfu:
			n, c, h, w = x.shape
			split_no = 2
			split_s = h // split_no
			xs = torch.cat(torch.split(
				x[:, :c // 4], split_s, dim=-2), dim=1).contiguous()
			xs = torch.cat(torch.split(xs, split_s, dim=-1),
						   dim=1).contiguous()
			xs = self.lfu(xs)
			xs = xs.repeat(1, 1, split_no, split_no).contiguous()
		else:
			xs = 0

		output = self.conv2(x + output + xs)

		return output


class FFC(nn.Module):

	def __init__(self, in_channels, out_channels, kernel_size,
				 ratio_gin, ratio_gout, stride=1, padding=0,
				 dilation=1, groups=1, bias=False, enable_lfu=True,
				 padding_type='reflect', gated=False, **spectral_kwargs):
		super(FFC, self).__init__()

		assert stride == 1 or stride == 2, "Stride should be 1 or 2."
		self.stride = stride

		in_cg = int(in_channels * ratio_gin)
		in_cl = in_channels - in_cg
		out_cg = int(out_channels * ratio_gout)
		out_cl = out_channels - out_cg
		#groups_g = 1 if groups == 1 else int(groups * ratio_gout)
		#groups_l = 1 if groups == 1 else groups - groups_g

		self.ratio_gin = ratio_gin
		self.ratio_gout = ratio_gout
		self.global_in_num = in_cg

		module = nn.Identity if in_cl == 0 or out_cl == 0 else nn.Conv2d
		self.convl2l = module(in_cl, out_cl, kernel_size,
							  stride, padding, dilation, groups, bias, padding_mode=padding_type)
		module = nn.Identity if in_cl == 0 or out_cg == 0 else nn.Conv2d
		self.convl2g = module(in_cl, out_cg, kernel_size,
							  stride, padding, dilation, groups, bias, padding_mode=padding_type)
		module = nn.Identity if in_cg == 0 or out_cl == 0 else nn.Conv2d
		self.convg2l = module(in_cg, out_cl, kernel_size,
							  stride, padding, dilation, groups, bias, padding_mode=padding_type)
		module = nn.Identity if in_cg == 0 or out_cg == 0 else SpectralTransform
		self.convg2g = module(
			in_cg, out_cg, stride, 1 if groups == 1 else groups // 2, enable_lfu, **spectral_kwargs)

		self.gated = gated
		module = nn.Identity if in_cg == 0 or out_cl == 0 or not self.gated else nn.Conv2d
		self.gate = module(in_channels, 2, 1)

	def forward(self, x):
		x_l, x_g = x if type(x) is tuple else (x, 0)
		out_xl, out_xg = 0, 0

		if self.gated:
			total_input_parts = [x_l]
			if torch.is_tensor(x_g):
				total_input_parts.append(x_g)
			total_input = torch.cat(total_input_parts, dim=1)

			gates = torch.sigmoid(self.gate(total_input))
			g2l_gate, l2g_gate = gates.chunk(2, dim=1)
		else:
			g2l_gate, l2g_gate = 1, 1

		if self.ratio_gout != 1:
			out_xl = self.convl2l(x_l) + self.convg2l(x_g) * g2l_gate
		if self.ratio_gout != 0:
			out_xg = self.convl2g(x_l) * l2g_gate + self.convg2g(x_g)

		return out_xl, out_xg


class FFC_BN_ACT(nn.Module):

	def __init__(self, in_channels, out_channels,
				 kernel_size, ratio_gin, ratio_gout,
				 stride=1, padding=0, dilation=1, groups=1, bias=False,
				 norm_layer=nn.BatchNorm2d, activation_layer=nn.Identity,
				 padding_type='reflect',
				 enable_lfu=True, **kwargs):
		super(FFC_BN_ACT, self).__init__()
		self.ffc = FFC(in_channels, out_channels, kernel_size,
					   ratio_gin, ratio_gout, stride, padding, dilation,
					   groups, bias, enable_lfu, padding_type=padding_type, **kwargs)
		lnorm = nn.Identity if ratio_gout == 1 else norm_layer
		gnorm = nn.Identity if ratio_gout == 0 else norm_layer
		global_channels = int(out_channels * ratio_gout)
		self.bn_l = lnorm(out_channels - global_channels)
		self.bn_g = gnorm(global_channels)

		lact = nn.Identity if ratio_gout == 1 else activation_layer
		gact = nn.Identity if ratio_gout == 0 else activation_layer
		self.act_l = lact(inplace=True)
		self.act_g = gact(inplace=True)

	def forward(self, x):
		x_l, x_g = self.ffc(x)
		x_l = self.act_l(self.bn_l(x_l))
		x_g = self.act_g(self.bn_g(x_g))
		return x_l, x_g


class FFCResnetBlock(nn.Module):
	def __init__(self, dim, padding_type, norm_layer, activation_layer=nn.ReLU, dilation=1,
				 spatial_transform_kwargs=None, inline=False, **conv_kwargs):
		super().__init__()
		self.conv1 = FFC_BN_ACT(dim, dim, kernel_size=3, padding=dilation, dilation=dilation,
								norm_layer=norm_layer,
								activation_layer=activation_layer,
								padding_type=padding_type,
								**conv_kwargs)
		self.conv2 = FFC_BN_ACT(dim, dim, kernel_size=3, padding=dilation, dilation=dilation,
								norm_layer=norm_layer,
								activation_layer=activation_layer,
								padding_type=padding_type,
								**conv_kwargs)
		if spatial_transform_kwargs is not None:
			self.conv1 = LearnableSpatialTransformWrapper(self.conv1, **spatial_transform_kwargs)
			self.conv2 = LearnableSpatialTransformWrapper(self.conv2, **spatial_transform_kwargs)
		self.inline = inline

	def forward(self, x):
		if self.inline:
			x_l, x_g = x[:, :-self.conv1.ffc.global_in_num], x[:, -self.conv1.ffc.global_in_num:]
		else:
			x_l, x_g = x if type(x) is tuple else (x, 0)

		id_l, id_g = x_l, x_g

		x_l, x_g = self.conv1((x_l, x_g))
		x_l, x_g = self.conv2((x_l, x_g))

		x_l, x_g = id_l + x_l, id_g + x_g
		out = x_l, x_g
		if self.inline:
			out = torch.cat(out, dim=1)
		return out


class ConcatTupleLayer(nn.Module):
	def forward(self, x):
		assert isinstance(x, tuple)
		x_l, x_g = x
		assert torch.is_tensor(x_l) or torch.is_tensor(x_g)
		if not torch.is_tensor(x_g):
			return x_l
		return torch.cat(x, dim=1)


class FFCResNetGenerator(nn.Module):
	def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d,
				 padding_type='reflect', activation_layer=nn.ReLU,
				 up_norm_layer=nn.BatchNorm2d, up_activation=nn.ReLU(True),
				 init_conv_kwargs={}, downsample_conv_kwargs={}, resnet_conv_kwargs={},
				 spatial_transform_layers=None, spatial_transform_kwargs={},
				 add_out_act=True, max_features=1024, out_ffc=False, out_ffc_kwargs={}):
		assert (n_blocks >= 0)
		super().__init__()

		model = [nn.ReflectionPad2d(1),
				 FFC_BN_ACT(input_nc, ngf, kernel_size=3, padding=0, norm_layer=norm_layer,
							activation_layer=activation_layer, **init_conv_kwargs)]

		### downsample
		for i in range(n_downsampling):
			mult = 2 ** i
			if i == n_downsampling - 1:
				cur_conv_kwargs = dict(downsample_conv_kwargs)
				cur_conv_kwargs['ratio_gout'] = resnet_conv_kwargs.get('ratio_gin', 0)
			else:
				cur_conv_kwargs = downsample_conv_kwargs
			model += [FFC_BN_ACT(min(max_features, ngf * mult),
								 min(max_features, ngf * mult * 2),
								 kernel_size=4, stride=2, padding=1,
								 norm_layer=norm_layer,
								 activation_layer=activation_layer,
								 **cur_conv_kwargs)]

		mult = 2 ** n_downsampling
		feats_num_bottleneck = min(max_features, ngf * mult)

		### resnet blocks
		for i in range(n_blocks):
			cur_resblock = FFCResnetBlock(feats_num_bottleneck, padding_type=padding_type, activation_layer=activation_layer,
										  norm_layer=norm_layer, **resnet_conv_kwargs)
			if spatial_transform_layers is not None and i in spatial_transform_layers:
				cur_resblock = LearnableSpatialTransformWrapper(cur_resblock, **spatial_transform_kwargs)
			model += [cur_resblock]

		model += [ConcatTupleLayer()]

		### upsample
		for i in range(n_downsampling):
			mult = 2 ** (n_downsampling - i)
			model += [nn.ConvTranspose2d(min(max_features, ngf * mult),
										 min(max_features, int(ngf * mult / 2)),
										 kernel_size=4, stride=2, padding=1, output_padding=0),
					  up_norm_layer(min(max_features, int(ngf * mult / 2))),
					  up_activation]

		if out_ffc:
			model += [FFCResnetBlock(ngf, padding_type=padding_type, activation_layer=activation_layer,
									 norm_layer=norm_layer, inline=True, **out_ffc_kwargs)]

		model += [nn.ReflectionPad2d(1),
				  nn.Conv2d(ngf, output_nc, kernel_size=3, padding=0)]
		if add_out_act:
			model.append(get_activation('tanh' if add_out_act is True else add_out_act))
		self.model = nn.Sequential(*model)

	def forward(self, input, mask = None):
		if mask is not None :
			input = torch.cat([mask, input], dim = 1)
		return self.model(input)


class FFCNLayerDiscriminator(BaseDiscriminator):
	def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, max_features=512,
				 init_conv_kwargs={}, conv_kwargs={}):
		super().__init__()
		self.n_layers = n_layers

		def _act_ctor(inplace=True):
			return nn.LeakyReLU(negative_slope=0.2, inplace=inplace)

		kw = 3
		padw = int(np.ceil((kw-1.0)/2))
		sequence = [[FFC_BN_ACT(input_nc, ndf, kernel_size=kw, padding=padw, norm_layer=norm_layer,
								activation_layer=_act_ctor, **init_conv_kwargs)]]

		nf = ndf
		for n in range(1, n_layers):
			nf_prev = nf
			nf = min(nf * 2, max_features)

			cur_model = [
				FFC_BN_ACT(nf_prev, nf,
						   kernel_size=kw, stride=2, padding=padw,
						   norm_layer=norm_layer,
						   activation_layer=_act_ctor,
						   **conv_kwargs)
			]
			sequence.append(cur_model)

		nf_prev = nf
		nf = min(nf * 2, 512)

		cur_model = [
			FFC_BN_ACT(nf_prev, nf,
					   kernel_size=kw, stride=1, padding=padw,
					   norm_layer=norm_layer,
					   activation_layer=lambda *args, **kwargs: nn.LeakyReLU(*args, negative_slope=0.2, **kwargs),
					   **conv_kwargs),
			ConcatTupleLayer()
		]
		sequence.append(cur_model)

		sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

		for n in range(len(sequence)):
			setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))

	def get_all_activations(self, x):
		res = [x]
		for n in range(self.n_layers + 2):
			model = getattr(self, 'model' + str(n))
			res.append(model(res[-1]))
		return res[1:]

	def forward(self, x):
		act = self.get_all_activations(x)
		feats = []
		for out in act[:-1]:
			if isinstance(out, tuple):
				if torch.is_tensor(out[1]):
					out = torch.cat(out, dim=1)
				else:
					out = out[0]
			feats.append(out)
		return act[-1], feats

def get_generator(n_blocks: int = 9) :
	init_conv_kwargs = {
		'ratio_gin': 0,
		'ratio_gout': 0,
		'enable_lfu': False,
	}
	downsample_conv_kwargs = {
		'ratio_gin': 0,
		'ratio_gout': 0,
		'enable_lfu': False,
	}
	resnet_conv_kwargs = {
		'ratio_gin': 0.75,
		'ratio_gout': 0.75,
		'enable_lfu': False,
	}
	return FFCResNetGenerator(4, 3, ngf=64, n_blocks=n_blocks,add_out_act=False,init_conv_kwargs=init_conv_kwargs,downsample_conv_kwargs=downsample_conv_kwargs,resnet_conv_kwargs=resnet_conv_kwargs)

def get_discriminator() :
	init_conv_kwargs = {
		'ratio_gin': 0,
		'ratio_gout': 0,
		'enable_lfu': False,
	}
	conv_kwargs = {
		'ratio_gin': 0,
		'ratio_gout': 0,
		'enable_lfu': False,
	}
	return FFCNLayerDiscriminator(3, norm_layer = nn.Identity, init_conv_kwargs = init_conv_kwargs, conv_kwargs = conv_kwargs)

from torchsummary import summary

def test_model() :
	dis = get_generator()
	image = torch.randn((1, 4, 640, 640))
	final = dis(image)
	breakpoint()
	print(final.shape)


if __name__ == '__main__' :
	test_model()
