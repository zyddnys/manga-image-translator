
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
								 3], kernel_size=2, stride=(2, 1), padding=(0, 1), bias=False)
		self.bn4_2 = nn.BatchNorm2d(self.output_channel_block[3])
		self.conv4_2 = nn.Conv2d(self.output_channel_block[3], self.output_channel_block[
								 3], kernel_size=2, stride=1, padding=0, bias=False)
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
		self.ConvNet = ResNet(input_channel, output_channel, BasicBlock, [3, 6, 7, 5])

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
		x = x + self.pe[offset: offset + x.size(0), :]
		return x#self.dropout(x)

def generate_square_subsequent_mask(sz):
	mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
	mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
	return mask

class AddCoords(nn.Module):

	def __init__(self, with_r=False):
		super().__init__()
		self.with_r = with_r

	def forward(self, input_tensor):
		"""
		Args:
			input_tensor: shape(batch, channel, x_dim, y_dim)
		"""
		batch_size, _, x_dim, y_dim = input_tensor.size()

		xx_channel = torch.arange(x_dim).repeat(1, y_dim, 1)
		yy_channel = torch.arange(y_dim).repeat(1, x_dim, 1).transpose(1, 2)

		xx_channel = xx_channel.float() / (x_dim - 1)
		yy_channel = yy_channel.float() / (y_dim - 1)

		xx_channel = xx_channel * 2 - 1
		yy_channel = yy_channel * 2 - 1

		xx_channel = xx_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
		yy_channel = yy_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)

		ret = torch.cat([
			input_tensor,
			xx_channel.type_as(input_tensor),
			yy_channel.type_as(input_tensor)], dim=1)

		if self.with_r:
			rr = torch.sqrt(torch.pow(xx_channel.type_as(input_tensor) - 0.5, 2) + torch.pow(yy_channel.type_as(input_tensor) - 0.5, 2))
			ret = torch.cat([ret, rr], dim=1)

		return ret

class Beam :
	def __init__(self, char_seq = [], logprobs = []) :
		# L
		if isinstance(char_seq, list) :
			self.chars = torch.tensor(char_seq, dtype=torch.long)
			self.logprobs = torch.tensor(logprobs, dtype=torch.float32)
		else :
			self.chars = char_seq.clone()
			self.logprobs = logprobs.clone()

	def avg_logprob(self) :
		return self.logprobs.mean().item()
	
	def sort_key(self) :
		return -self.avg_logprob()
	
	def seq_end(self, end_tok) :
		return self.chars.view(-1)[-1] == end_tok

	def extend(self, idx, logprob) :
		return Beam(
			torch.cat([self.chars, idx.unsqueeze(0)], dim = -1),
			torch.cat([self.logprobs, logprob.unsqueeze(0)], dim = -1),
		)

DECODE_BLOCK_LENGTH = 8

class Hypothesis :
	def __init__(self, device, start_tok: int, end_tok: int, padding_tok: int, memory_idx: int, num_layers: int, embd_dim: int) :
		self.device = device
		self.start_tok = start_tok
		self.end_tok = end_tok
		self.padding_tok = padding_tok
		self.memory_idx = memory_idx
		self.embd_size = embd_dim
		self.num_layers = num_layers
		# L, 1, E
		self.cached_activations = [torch.zeros(0, 1, self.embd_size).to(self.device)] * (num_layers + 1)
		self.out_idx = torch.LongTensor([start_tok]).to(self.device)
		self.out_logprobs = torch.FloatTensor([0]).to(self.device)
		self.length = 0

	def seq_end(self) :
		return self.out_idx.view(-1)[-1] == self.end_tok

	def logprob(self) :
		return self.out_logprobs.mean().item()

	def sort_key(self) :
		return -self.logprob()

	def prob(self) :
		return self.out_logprobs.mean().exp().item()

	def __len__(self) :
		return self.length

	def extend(self, idx, logprob) :
		ret = Hypothesis(self.device, self.start_tok, self.end_tok, self.padding_tok, self.memory_idx, self.num_layers, self.embd_size)
		ret.cached_activations = [item.clone() for item in self.cached_activations]
		ret.length = self.length + 1
		ret.out_idx = torch.cat([self.out_idx, torch.LongTensor([idx]).to(self.device)], dim = 0)
		ret.out_logprobs = torch.cat([self.out_logprobs, torch.FloatTensor([logprob]).to(self.device)], dim = 0)
		return ret

	def output(self) :
		return self.cached_activations[-1]

def next_token_batch(
	hyps: List[Hypothesis],
	memory: torch.Tensor, # S, K, E
	memory_mask: torch.BoolTensor,
	decoders: nn.TransformerDecoder,
	pe: PositionalEncoding,
	embd: nn.Embedding
	) :
	layer: nn.TransformerDecoderLayer
	N = len(hyps)

	# N
	last_toks = torch.stack([item.out_idx[-1] for item in hyps], dim = 0)
	# 1, N, E
	tgt: torch.FloatTensor = pe(embd(last_toks).unsqueeze_(0), offset = len(hyps[0]))

	# # L, N
	# out_idxs = torch.stack([item.out_idx for item in hyps], dim = 0).permute(1, 0)
	# # L, N, E
	# tgt2: torch.FloatTensor = pe(embd(out_idxs))
	# # 1, N, E
	# tgt_v2 = tgt2[-1, :, :].unsqueeze_(0)
	# print(((tgt_v1 - tgt_v2) ** 2).sum())

	# tgt = tgt_v2

	# S, N, E
	memory = torch.stack([memory[:, idx, :] for idx in [item.memory_idx for item in hyps]], dim = 1)
	for l, layer in enumerate(decoders.layers) :
		# TODO: keys and values are recomputed everytime
		# L - 1, N, E
		combined_activations = torch.cat([item.cached_activations[l] for item in hyps], dim = 1)
		# L, N, E
		combined_activations = torch.cat([combined_activations, tgt], dim = 0)
		for i in range(N) :
			hyps[i].cached_activations[l] = combined_activations[:, i: i + 1, :]
		tgt2 = layer.self_attn(tgt, combined_activations, combined_activations)[0]
		tgt = tgt + layer.dropout1(tgt2)
		tgt = layer.norm1(tgt)
		tgt2 = layer.multihead_attn(tgt, memory, memory, key_padding_mask = memory_mask)[0]
		tgt = tgt + layer.dropout2(tgt2)
		tgt = layer.norm2(tgt)
		tgt2 = layer.linear2(layer.dropout(layer.activation(layer.linear1(tgt))))
		tgt = tgt + layer.dropout3(tgt2)
		# 1, N, E
		tgt = layer.norm3(tgt)
	#print(tgt[0, 0, 0])
	for i in range(N) :
		hyps[i].cached_activations[decoders.num_layers] = torch.cat([hyps[i].cached_activations[decoders.num_layers], tgt[:, i: i + 1, :]], dim = 0)
	# N, E
	return tgt.squeeze_(0)

class OCR(nn.Module) :
	def __init__(self, dictionary, max_len):
		super(OCR, self).__init__()
		self.max_len = max_len
		self.dictionary = dictionary
		self.dict_size = len(dictionary)
		self.backbone = ResNet_FeatureExtractor(3, 320)
		encoder = nn.TransformerEncoderLayer(320, 4, dropout = 0.0)
		decoder = nn.TransformerDecoderLayer(320, 4, dropout = 0.0)
		self.encoders = nn.TransformerEncoder(encoder, 3)
		self.decoders = nn.TransformerDecoder(decoder, 3)
		self.pe = PositionalEncoding(320, max_len = max_len)
		self.embd = nn.Embedding(self.dict_size, 320)
		self.pred1 = nn.Sequential(nn.Linear(320, 320), nn.ReLU())
		self.pred = nn.Linear(320, self.dict_size)
		self.pred.weight = self.embd.weight
		self.color_pred1 = nn.Sequential(nn.Linear(320, 64), nn.ReLU())
		self.fg_r_pred = nn.Linear(64, 1)
		self.fg_g_pred = nn.Linear(64, 1)
		self.fg_b_pred = nn.Linear(64, 1)
		self.bg_r_pred = nn.Linear(64, 1)
		self.bg_g_pred = nn.Linear(64, 1)
		self.bg_b_pred = nn.Linear(64, 1)

	def forward(self,
		img: torch.FloatTensor,
		char_idx: torch.LongTensor,
		mask: torch.BoolTensor,
		source_mask: torch.BoolTensor
		) :
		feats = self.backbone(img)
		feats = torch.einsum('n e h s -> s n e', feats)
		feats = self.pe(feats)
		memory = self.encoders(feats, src_key_padding_mask = source_mask)
		N, L = char_idx.shape
		char_embd = self.embd(char_idx)
		char_embd = torch.einsum('n t e -> t n e', char_embd)
		char_embd = self.pe(char_embd)
		casual_mask = generate_square_subsequent_mask(L).to(img.device)
		decoded = self.decoders(char_embd, memory, tgt_mask = casual_mask, tgt_key_padding_mask = mask, memory_key_padding_mask = source_mask)
		decoded = decoded.permute(1, 0, 2)
		pred_char_logits = self.pred(self.pred1(decoded))
		color_feats = self.color_pred1(decoded)
		return pred_char_logits, \
			self.fg_r_pred(color_feats), \
			self.fg_g_pred(color_feats), \
			self.fg_b_pred(color_feats), \
			self.bg_r_pred(color_feats), \
			self.bg_g_pred(color_feats), \
			self.bg_b_pred(color_feats)

	def infer_beam_batch(self, img: torch.FloatTensor, img_widths: List[int], beams_k: int = 5, start_tok = 1, end_tok = 2, pad_tok = 0, max_finished_hypos: int = 2, max_seq_length = 384) :
		N, C, H, W = img.shape
		assert H == 32 and C == 3
		feats = self.backbone(img)
		feats = torch.einsum('n e h s -> s n e', feats)
		valid_feats_length = [(x + 3) // 4 + 2 for x in img_widths]
		input_mask = torch.zeros(N, feats.size(0), dtype = torch.bool).to(img.device)
		for i, l in enumerate(valid_feats_length) :
			input_mask[i, l:] = True
		feats = self.pe(feats)
		memory = self.encoders(feats, src_key_padding_mask = input_mask)
		hypos = [Hypothesis(img.device, start_tok, end_tok, pad_tok, i, self.decoders.num_layers, 320) for i in range(N)]
		# N, E
		decoded = next_token_batch(hypos, memory, input_mask, self.decoders, self.pe, self.embd)
		# N, n_chars
		pred_char_logprob = self.pred(self.pred1(decoded)).log_softmax(-1)
		# N, k
		pred_chars_values, pred_chars_index = torch.topk(pred_char_logprob, beams_k, dim = 1)
		new_hypos = []
		finished_hypos = defaultdict(list)
		for i in range(N) :
			for k in range(beams_k) :
				new_hypos.append(hypos[i].extend(pred_chars_index[i, k], pred_chars_values[i, k]))
		hypos = new_hypos
		for _ in range(max_seq_length) :
			# N * k, E
			decoded = next_token_batch(hypos, memory, torch.stack([input_mask[hyp.memory_idx] for hyp in hypos]) , self.decoders, self.pe, self.embd)
			# N * k, n_chars
			pred_char_logprob = self.pred(self.pred1(decoded)).log_softmax(-1)
			# N * k, k
			pred_chars_values, pred_chars_index = torch.topk(pred_char_logprob, beams_k, dim = 1)
			hypos_per_sample = defaultdict(list)
			h: Hypothesis
			for i, h in enumerate(hypos) :
				for k in range(beams_k) :
					hypos_per_sample[h.memory_idx].append(h.extend(pred_chars_index[i, k], pred_chars_values[i, k]))
			hypos = []
			# hypos_per_sample now contains N * k^2 hypos
			for i in hypos_per_sample.keys() :
				cur_hypos: List[Hypothesis] = hypos_per_sample[i]
				cur_hypos = sorted(cur_hypos, key = lambda a: a.sort_key())[: beams_k + 1]
				#print(cur_hypos[0].out_idx[-1])
				to_added_hypos = []
				sample_done = False
				for h in cur_hypos :
					if h.seq_end() :
						finished_hypos[i].append(h)
						if len(finished_hypos[i]) >= max_finished_hypos :
							sample_done = True
							break
					else :
						if len(to_added_hypos) < beams_k :
							to_added_hypos.append(h)
				if not sample_done :
					hypos.extend(to_added_hypos)
			if len(hypos) == 0 :
				break
		# add remaining hypos to finished
		for i in range(N) :
			if i not in finished_hypos :
				cur_hypos: List[Hypothesis] = hypos_per_sample[i]
				cur_hypo = sorted(cur_hypos, key = lambda a: a.sort_key())[0]
				finished_hypos[i].append(cur_hypo)
		assert len(finished_hypos) == N
		result = []
		for i in range(N) :
			cur_hypos = finished_hypos[i]
			cur_hypo = sorted(cur_hypos, key = lambda a: a.sort_key())[0]
			decoded = cur_hypo.output()
			color_feats = self.color_pred1(decoded)
			fg_r, fg_g, fg_b, bg_r, bg_g, bg_b = self.fg_r_pred(color_feats), \
				self.fg_g_pred(color_feats), \
				self.fg_b_pred(color_feats), \
				self.bg_r_pred(color_feats), \
				self.bg_g_pred(color_feats), \
				self.bg_b_pred(color_feats)
			result.append((cur_hypo.out_idx, cur_hypo.prob(), fg_r, fg_g, fg_b, bg_r, bg_g, bg_b))
		return result

	def infer_beam(self, img: torch.FloatTensor, beams_k: int = 5, start_tok = 1, end_tok = 2, pad_tok = 0, max_seq_length = 384) :
		N, C, H, W = img.shape
		assert H == 32 and N == 1 and C == 3
		feats = self.backbone(img)
		feats = torch.einsum('n e h s -> s n e', feats)
		feats = self.pe(feats)
		memory = self.encoders(feats)
		def run(tokens, add_start_tok = True, char_only = True) :
			if add_start_tok :
				if isinstance(tokens, list) :
					# N(=1), L
					tokens = torch.tensor([start_tok] + tokens, dtype = torch.long, device = img.device).unsqueeze_(0)
				else :
					# N, L
					tokens = torch.cat([torch.tensor([start_tok], dtype = torch.long, device = img.device), tokens], dim = -1).unsqueeze_(0)
			N, L = tokens.shape
			embd = self.embd(tokens)
			embd = torch.einsum('n t e -> t n e', embd)
			embd = self.pe(embd)
			casual_mask = generate_square_subsequent_mask(L).to(img.device)
			decoded = self.decoders(embd, memory, tgt_mask = casual_mask)
			decoded = decoded.permute(1, 0, 2)
			pred_char_logprob = self.pred(self.pred1(decoded)).log_softmax(-1)
			if char_only :
				return pred_char_logprob
			else :
				color_feats = self.color_pred1(decoded)
				return pred_char_logprob, \
					self.fg_r_pred(color_feats), \
					self.fg_g_pred(color_feats), \
					self.fg_b_pred(color_feats), \
					self.bg_r_pred(color_feats), \
					self.bg_g_pred(color_feats), \
					self.bg_b_pred(color_feats)
		# N, L, embd_size
		initial_char_logprob = run([])
		# N, L
		initial_pred_chars_values, initial_pred_chars_index = torch.topk(initial_char_logprob, beams_k, dim = 2)
		# beams_k, L
		initial_pred_chars_values = initial_pred_chars_values.squeeze(0).permute(1, 0)
		initial_pred_chars_index = initial_pred_chars_index.squeeze(0).permute(1, 0)
		beams = sorted([Beam(tok, logprob) for tok, logprob in zip(initial_pred_chars_index, initial_pred_chars_values)], key = lambda a: a.sort_key())
		for _ in range(max_seq_length) :
			new_beams = []
			all_ended = True
			for beam in beams :
				if not beam.seq_end(end_tok) :
					logprobs = run(beam.chars)
					pred_chars_values, pred_chars_index = torch.topk(logprobs, beams_k, dim = 2)
					# beams_k, L
					pred_chars_values = pred_chars_values.squeeze(0).permute(1, 0)
					pred_chars_index = pred_chars_index.squeeze(0).permute(1, 0)
					#print(pred_chars_index.view(-1)[-1])
					new_beams.extend([beam.extend(tok[-1], logprob[-1]) for tok, logprob in zip(pred_chars_index, pred_chars_values)])
					#new_beams.extend([Beam(tok, logprob) for tok, logprob in zip(pred_chars_index, pred_chars_values)]) # extend other top k
					all_ended = False
				else :
					new_beams.append(beam) # seq ended, add back to queue
			beams = sorted(new_beams, key = lambda a: a.sort_key())[: beams_k] # keep top k
			#print(beams[0].chars)
			if all_ended :
				break
		final_tokens = beams[0].chars[:-1]
		#print(beams[0].logprobs.mean().exp())
		return run(final_tokens, char_only = False), beams[0].logprobs.mean().exp().item()

def test() :
	with open('../SynthText/alphabet-all-v2.txt', 'r') as fp :
		dictionary = [s[:-1] for s in fp.readlines()]
	img = torch.randn(4, 3, 32, 1224)
	idx = torch.zeros(4, 32).long()
	mask = torch.zeros(4, 32).bool()
	model = ResNet_FeatureExtractor(3, 256)
	out = model(img)

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
	test()
