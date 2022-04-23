
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F

import heapq
import math
import einops

from typing import List, Tuple, Optional

class PositionalEncoding(nn.Module):
	def __init__(self, d_model, dropout=0.1, max_len=5000):
		super(PositionalEncoding, self).__init__()
		self.dropout = nn.Dropout(p=dropout)

		pe = torch.zeros(max_len, d_model)
		position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)
		pe = pe.unsqueeze(0)
		self.register_buffer('pe', pe)

	def forward(self, x, offset = 0):
		x = x + self.pe[:, offset: offset + x.size(1), :]
		return x

class CustomTransformerEncoderLayer(nn.Module):
	r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
	This standard encoder layer is based on the paper "Attention Is All You Need".
	Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
	Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
	Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
	in a different way during application.

	Args:
		d_model: the number of expected features in the input (required).
		nhead: the number of heads in the multiheadattention models (required).
		dim_feedforward: the dimension of the feedforward network model (default=2048).
		dropout: the dropout value (default=0.1).
		activation: the activation function of intermediate layer, relu or gelu (default=relu).
		layer_norm_eps: the eps value in layer normalization components (default=1e-5).
		batch_first: If ``True``, then the input and output tensors are provided
			as (batch, seq, feature). Default: ``False``.
		norm_first: if ``True``, layer norm is done prior to attention and feedforward
			operations, respectivaly. Otherwise it's done after. Default: ``False`` (after).

	Examples::
		>>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
		>>> src = torch.rand(10, 32, 512)
		>>> out = encoder_layer(src)

	Alternatively, when ``batch_first`` is ``True``:
		>>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
		>>> src = torch.rand(32, 10, 512)
		>>> out = encoder_layer(src)
	"""
	__constants__ = ['batch_first', 'norm_first']

	def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="gelu",
				 layer_norm_eps=1e-5, batch_first=False, norm_first=False,
				 device=None, dtype=None) -> None:
		factory_kwargs = {'device': device, 'dtype': dtype}
		super(CustomTransformerEncoderLayer, self).__init__()
		self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
											**factory_kwargs)
		# Implementation of Feedforward model
		self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
		self.dropout = nn.Dropout(dropout)
		self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

		self.norm_first = norm_first
		self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
		self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
		self.dropout1 = nn.Dropout(dropout)
		self.dropout2 = nn.Dropout(dropout)
		self.pe = PositionalEncoding(d_model, max_len = 768)

		self.activation = F.gelu

	def __setstate__(self, state):
		if 'activation' not in state:
			state['activation'] = F.relu
		super(CustomTransformerEncoderLayer, self).__setstate__(state)

	def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None, src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
		r"""Pass the input through the encoder layer.

		Args:
			src: the sequence to the encoder layer (required).
			src_mask: the mask for the src sequence (optional).
			src_key_padding_mask: the mask for the src keys per batch (optional).

		Shape:
			see the docs in Transformer class.
		"""

		# see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

		x = src
		if self.norm_first:
			x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
			x = x + self._ff_block(self.norm2(x))
		else:
			x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
			x = self.norm2(x + self._ff_block(x))

		return x

	# self-attention block
	def _sa_block(self, x: torch.Tensor,
				  attn_mask: Optional[torch.Tensor], key_padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
		x = self.self_attn(self.pe(x), self.pe(x), x, # no PE for value
						   attn_mask=attn_mask,
						   key_padding_mask=key_padding_mask,
						   need_weights=False)[0]
		return self.dropout1(x)

	# feed forward block
	def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
		x = self.linear2(self.dropout(self.activation(self.linear1(x))))
		return self.dropout2(x)


class ResNet(nn.Module):

	def __init__(self, input_channel, output_channel, block, layers):
		super(ResNet, self).__init__()

		self.output_channel_block = [int(output_channel / 4), int(output_channel / 2), output_channel, output_channel]

		self.inplanes = int(output_channel / 8)
		self.conv0_1 = nn.Conv2d(input_channel, int(output_channel / 8),
								 kernel_size=3, stride=1, padding=1, bias=False)
		self.bn0_1 = nn.BatchNorm2d(int(output_channel / 8))
		self.conv0_2 = nn.Conv2d(int(output_channel / 8), self.inplanes,
								 kernel_size=3, stride=1, padding=1, bias=False)

		self.maxpool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
		self.layer1 = self._make_layer(block, self.output_channel_block[0], layers[0])
		self.bn1 = nn.BatchNorm2d(self.output_channel_block[0])
		self.conv1 = nn.Conv2d(self.output_channel_block[0], self.output_channel_block[
							   0], kernel_size=3, stride=1, padding=1, bias=False)

		self.maxpool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
		self.layer2 = self._make_layer(block, self.output_channel_block[1], layers[1], stride=1)
		self.bn2 = nn.BatchNorm2d(self.output_channel_block[1])
		self.conv2 = nn.Conv2d(self.output_channel_block[1], self.output_channel_block[
							   1], kernel_size=3, stride=1, padding=1, bias=False)

		self.maxpool3 = nn.AvgPool2d(kernel_size=2, stride=(2, 1), padding=(0, 1))
		self.layer3 = self._make_layer(block, self.output_channel_block[2], layers[2], stride=1)
		self.bn3 = nn.BatchNorm2d(self.output_channel_block[2])
		self.conv3 = nn.Conv2d(self.output_channel_block[2], self.output_channel_block[
							   2], kernel_size=3, stride=1, padding=1, bias=False)

		self.layer4 = self._make_layer(block, self.output_channel_block[3], layers[3], stride=1)
		self.bn4_1 = nn.BatchNorm2d(self.output_channel_block[3])
		self.conv4_1 = nn.Conv2d(self.output_channel_block[3], self.output_channel_block[
								 3], kernel_size=3, stride=(2, 1), padding=(1, 1), bias=False)
		self.bn4_2 = nn.BatchNorm2d(self.output_channel_block[3])
		self.conv4_2 = nn.Conv2d(self.output_channel_block[3], self.output_channel_block[
								 3], kernel_size=3, stride=1, padding=0, bias=False)
		self.bn4_3 = nn.BatchNorm2d(self.output_channel_block[3])

	def _make_layer(self, block, planes, blocks, stride=1):
		downsample = None
		if stride != 1 or self.inplanes != planes * block.expansion:
			downsample = nn.Sequential(
				nn.BatchNorm2d(self.inplanes),
				nn.Conv2d(self.inplanes, planes * block.expansion,
						  kernel_size=1, stride=stride, bias=False),
			)

		layers = []
		layers.append(block(self.inplanes, planes, stride, downsample))
		self.inplanes = planes * block.expansion
		for i in range(1, blocks):
			layers.append(block(self.inplanes, planes))

		return nn.Sequential(*layers)

	def forward(self, x):
		x = self.conv0_1(x)
		x = self.bn0_1(x)
		x = F.relu(x)
		x = self.conv0_2(x)

		x = self.maxpool1(x)
		x = self.layer1(x)
		x = self.bn1(x)
		x = F.relu(x)
		x = self.conv1(x)

		x = self.maxpool2(x)
		x = self.layer2(x)
		x = self.bn2(x)
		x = F.relu(x)
		x = self.conv2(x)

		x = self.maxpool3(x)
		x = self.layer3(x)
		x = self.bn3(x)
		x = F.relu(x)
		x = self.conv3(x)

		x = self.layer4(x)


		x = self.bn4_1(x)
		x = F.relu(x)
		x = self.conv4_1(x)
		x = self.bn4_2(x)
		x = F.relu(x)
		x = self.conv4_2(x)
		x = self.bn4_3(x)

		return x

class BasicBlock(nn.Module):
	expansion = 1

	def __init__(self, inplanes, planes, stride=1, downsample=None):
		super(BasicBlock, self).__init__()
		self.bn1 = nn.BatchNorm2d(inplanes)
		self.conv1 = self._conv3x3(inplanes, planes)
		self.bn2 = nn.BatchNorm2d(planes)
		self.conv2 = self._conv3x3(planes, planes)
		self.downsample = downsample
		self.stride = stride

	def _conv3x3(self, in_planes, out_planes, stride=1):
		"3x3 convolution with padding"
		return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
						 padding=1, bias=False)

	def forward(self, x):
		residual = x

		out = self.bn1(x)
		out = F.relu(out)
		out = self.conv1(out)

		out = self.bn2(out)
		out = F.relu(out)
		out = self.conv2(out)

		if self.downsample is not None:
			residual = self.downsample(residual)

		return out + residual

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
	"""3x3 convolution with padding"""
	return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
					 padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
	"""1x1 convolution"""
	return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class ResNet_FeatureExtractor(nn.Module):
	""" FeatureExtractor of FAN (http://openaccess.thecvf.com/content_ICCV_2017/papers/Cheng_Focusing_Attention_Towards_ICCV_2017_paper.pdf) """

	def __init__(self, input_channel, output_channel=128):
		super(ResNet_FeatureExtractor, self).__init__()
		self.ConvNet = ResNet(input_channel, output_channel, BasicBlock, [4, 6, 8, 6, 3])

	def forward(self, input):
		return self.ConvNet(input)
		
class OCR(nn.Module) :
	def __init__(self, dictionary, max_len):
		super(OCR, self).__init__()
		self.max_len = max_len
		self.dictionary = dictionary
		self.dict_size = len(dictionary)
		self.backbone = ResNet_FeatureExtractor(3, 320)
		enc = CustomTransformerEncoderLayer(320, 8, 320 * 4, dropout=0.05,batch_first=True,norm_first=True)
		self.encoders = nn.TransformerEncoder(enc, 3)
		self.char_pred_norm = nn.Sequential(nn.LayerNorm(320), nn.Dropout(0.1), nn.GELU())
		self.char_pred = nn.Linear(320, self.dict_size)
		self.color_pred1 = nn.Sequential(nn.Linear(320, 6))

	def forward(self,
		img: torch.FloatTensor
		) :
		feats = self.backbone(img).squeeze(2)
		feats = self.encoders(feats.permute(0, 2, 1))
		pred_char_logits = self.char_pred(self.char_pred_norm(feats))
		pred_color_values = self.color_pred1(feats)
		return pred_char_logits, pred_color_values

	def decode(self, img: torch.Tensor, img_widths: List[int], blank, verbose = False) -> List[List[Tuple[str, float, int, int, int, int, int, int]]] :
		N, C, H, W = img.shape
		assert H == 48 and C == 3
		feats = self.backbone(img).squeeze(2)
		feats = self.encoders(feats.permute(0, 2, 1))
		pred_char_logits = self.char_pred(self.char_pred_norm(feats))
		pred_color_values = self.color_pred1(feats)
		return self.decode_ctc_top1(pred_char_logits, pred_color_values, blank, verbose = verbose)

	def decode_ctc_top1(self, pred_char_logits, pred_color_values, blank, verbose = False) -> List[List[Tuple[str, float, int, int, int, int, int, int]]] :
		pred_chars: List[List[Tuple[str, float, int, int, int, int, int, int]]] = []
		for _ in range(pred_char_logits.size(0)) :
			pred_chars.append([])
		logprobs = pred_char_logits.log_softmax(2)
		_, preds_index = logprobs.max(2)
		preds_index = preds_index.cpu()
		pred_color_values = pred_color_values.cpu().clamp_(0, 1)
		for b in range(pred_char_logits.size(0)) :
			if verbose :
				print('------------------------------')
			last_ch = blank
			for t in range(pred_char_logits.size(1)) :
				pred_ch = preds_index[b, t]
				if pred_ch != last_ch and pred_ch != blank :
					lp = logprobs[b, t, pred_ch].item()
					if verbose :
						if lp < math.log(0.9) :
							top5 = torch.topk(logprobs[b, t], 5)
							top5_idx = top5.indices
							top5_val = top5.values
							r = ''
							for i in range(5) :
								r += f'{self.dictionary[top5_idx[i]]}: {math.exp(top5_val[i])}, '
							print(r)
						else :
							print(f'{self.dictionary[pred_ch]}: {math.exp(lp)}')
					pred_chars[b].append((
						pred_ch,
						lp,
						pred_color_values[b, t][0].item(),
						pred_color_values[b, t][1].item(),
						pred_color_values[b, t][2].item(),
						pred_color_values[b, t][3].item(),
						pred_color_values[b, t][4].item(),
						pred_color_values[b, t][5].item()
					))
				last_ch = pred_ch
		return pred_chars

	def eval_ocr(self, input_lengths, target_lengths, pred_char_logits, pred_color_values, gt_char_index, gt_color_values, blank, blank1) :
		correct_char = 0
		total_char = 0
		color_diff = 0
		color_diff_dom = 0
		_, preds_index = pred_char_logits.max(2)
		pred_chars = torch.zeros_like(gt_char_index).cpu()
		for b in range(pred_char_logits.size(0)) :
			last_ch = blank
			i = 0
			for t in range(input_lengths[b]) :
				pred_ch = preds_index[b, t]
				if pred_ch != last_ch and pred_ch != blank :
					total_char += 1
					if gt_char_index[b, i] == pred_ch :
						correct_char += 1
						if pred_ch != blank1 :
							color_diff += ((pred_color_values[b, t] - gt_color_values[b, i]).abs().mean() * 255.0).item()
							color_diff_dom += 1
					pred_chars[b, i] = pred_ch
					i += 1
					if i >= gt_color_values.size(1) or i >= gt_char_index.size(1) :
						break
				last_ch = pred_ch
		return correct_char / (total_char + 1), color_diff / (color_diff_dom + 1), pred_chars

def test2() :
	with open('alphabet-all-v5.txt', 'r') as fp :
		dictionary = [s[:-1] for s in fp.readlines()]
	img = torch.randn(4, 3, 48, 1536)
	idx = torch.zeros(4, 32).long()
	mask = torch.zeros(4, 32).bool()
	model = OCR(dictionary, 1024)
	pred_char_logits, pred_color_values = model(img)
	print(pred_char_logits.shape, pred_color_values.shape)


def test_inference() :
	with torch.no_grad() :
		with open('../SynthText/alphabet-all-v3.txt', 'r') as fp :
			dictionary = [s[:-1] for s in fp.readlines()]
		img = torch.zeros(1, 3, 32, 128)
		model = OCR(dictionary, 32)
		m = torch.load("ocr_ar_v2-3-test.ckpt", map_location='cpu')
		model.load_state_dict(m['model'])
		model.eval()
		(char_probs, _, _, _, _, _, _, _), _ = model.infer_beam(img, max_seq_length = 20)
		_, pred_chars_index = char_probs.max(2)
		pred_chars_index = pred_chars_index.squeeze_(0)
		seq = []
		for chid in pred_chars_index :
			ch = dictionary[chid]
			if ch == '<SP>' :
				ch == ' '
			seq.append(ch)
		print(''.join(seq))

if __name__ == "__main__":
	test2()
