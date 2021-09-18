
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F

import heapq
import math
import einops

from typing import List, Tuple, Optional

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
		#self.layer5 = self._make_layer(block, self.output_channel_block[3], layers[4], stride=1)
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
		with torch.no_grad() :
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
		#x = self.layer5(x)
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

class PositionalEncoding(nn.Module):
	def __init__(self, d_model, dropout=0.1, max_len=5000):
		super(PositionalEncoding, self).__init__()
		self.dropout = nn.Dropout(p=dropout)

		pe = torch.zeros(max_len, d_model)
		position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)
		pe = pe.unsqueeze(0).transpose(0, 1)
		self.register_buffer('pe', pe)

	def forward(self, x, offset = 0):
		x = x + self.pe[offset: offset + x.size(0), :] / math.sqrt(x.size(2))
		return x

class OCR(nn.Module) :
	def __init__(self, dictionary, max_len):
		super(OCR, self).__init__()
		self.max_len = max_len
		self.dictionary = dictionary
		self.dict_size = len(dictionary)
		self.backbone = ResNet_FeatureExtractor(3, 320)
		encoder = nn.TransformerEncoderLayer(320, 4, dropout = 0.0)
		self.encoders = nn.TransformerEncoder(encoder, 3)
		self.pe = PositionalEncoding(320, max_len = max_len)
		self.char_pred = nn.Sequential(nn.Linear(320, 320), nn.ReLU(), nn.Linear(320, self.dict_size))
		self.color_pred1 = nn.Sequential(nn.Linear(320, 64), nn.ReLU(), nn.Linear(64, 6))

	def forward(self,
		img: torch.FloatTensor,
		source_mask: torch.BoolTensor
		) :
		feats = self.backbone(img)
		feats = torch.einsum('n e h s -> s n e', feats)
		feats = self.pe(feats)
		feats = self.encoders(feats, src_key_padding_mask = source_mask)
		feats = torch.einsum('s n e -> n s e', feats)
		pred_char_logits = self.char_pred(feats)
		pred_color_values = self.color_pred1(feats)
		return pred_char_logits, pred_color_values

	def decode(self, img: torch.Tensor, img_widths: List[int], blank) -> List[List[Tuple[str, float, int, int, int, int, int, int]]] :
		N, C, H, W = img.shape
		assert H == 48 and C == 3
		feats = self.backbone(img)
		feats = torch.einsum('n e h s -> s n e', feats)
		valid_feats_length = [(x + 3) // 4 + 1 for x in img_widths]
		input_mask = torch.zeros(N, feats.size(0), dtype = torch.bool).to(img.device)
		for i, l in enumerate(valid_feats_length) :
			input_mask[i, l:] = True
		feats = self.pe(feats)
		feats = self.encoders(feats, src_key_padding_mask = input_mask)
		feats = torch.einsum('s n e -> n s e', feats)
		pred_char_logits = self.char_pred(feats)
		pred_color_values = self.color_pred1(feats)
		return self.decode_ctc_top1(pred_char_logits, pred_color_values, blank)

	def decode_ctc_top1(self, pred_char_logits, pred_color_values, blank) -> List[List[Tuple[str, float, int, int, int, int, int, int]]] :
		pred_chars: List[List[Tuple[str, float, int, int, int, int, int, int]]] = []
		for _ in range(pred_char_logits.size(0)) :
			pred_chars.append([])
		logprobs = pred_char_logits.log_softmax(2)
		_, preds_index = logprobs.max(2)
		preds_index = preds_index.cpu()
		pred_color_values = pred_color_values.cpu()
		for b in range(pred_char_logits.size(0)) :
			last_ch = blank
			for t in range(pred_char_logits.size(1)) :
				pred_ch = preds_index[b, t]
				if pred_ch != last_ch and pred_ch != blank :
					lp = logprobs[b, t, pred_ch].item()
					if lp < math.log(0.96) :
						top5 = torch.topk(logprobs[b, t], 5)
						top5_idx = top5.indices
						top5_val = top5.values
						r = ''
						for i in range(5) :
							r += f'{self.dictionary[top5_idx[i]]}: {math.exp(top5_val[i])}, '
						print(r)
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
		return correct_char / (total_char+1), color_diff / (color_diff_dom+1), pred_chars
