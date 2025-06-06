
import math
from typing import Callable, List, Optional, Tuple, Union
from collections import defaultdict
import os
import shutil
import cv2
import numpy as np
import einops

import torch
import torch.nn as nn
import torch.nn.functional as F

from manga_translator.config import OcrConfig
from .xpos_relative_position import XPOS

# Roformer with Xpos and Local Attention ViT

from .common import OfflineOCR
from ..utils import TextBlock, Quadrilateral, chunks
from ..utils.generic import AvgMeter
from ..utils.bubble import is_ignore

# Roformer with Xpos

class Model48pxOCR(OfflineOCR):
    _MODEL_MAPPING = {
        'model': {
            'url': 'https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/ocr_ar_48px.ckpt',
            'hash': '29daa46d080818bb4ab239a518a88338cbccff8f901bef8c9db191a7cb97671d',
        },
        'dict': {
            'url': 'https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/alphabet-all-v7.txt',
            'hash': 'f5722368146aa0fbcc9f4726866e4efc3203318ebb66c811d8cbbe915576538a',
        },
    }

    def __init__(self, *args, **kwargs):
        os.makedirs(self.model_dir, exist_ok=True)
        if os.path.exists('ocr_ar_48px.ckpt'):
            shutil.move('ocr_ar_48px.ckpt', self._get_file_path('ocr_ar_48px.ckpt'))
        if os.path.exists('alphabet-all-v7.txt'):
            shutil.move('alphabet-all-v7.txt', self._get_file_path('alphabet-all-v7.txt'))
        super().__init__(*args, **kwargs)

    async def _load(self, device: str):
        with open(self._get_file_path('alphabet-all-v7.txt'), 'r', encoding = 'utf-8') as fp:
            dictionary = [s[:-1] for s in fp.readlines()]

        self.model = OCR(dictionary, 768)
        sd = torch.load(self._get_file_path('ocr_ar_48px.ckpt'))
        self.model.load_state_dict(sd)
        self.model.eval()
        self.device = device
        if (device == 'cuda' or device == 'mps'):
            self.use_gpu = True
        else:
            self.use_gpu = False
        if self.use_gpu:
            self.model = self.model.to(device)


    async def _unload(self):
        del self.model
    
    async def _infer(self, image: np.ndarray, textlines: List[Quadrilateral], config: OcrConfig, verbose: bool = False, ignore_bubble: int = 0) -> List[TextBlock]:
        text_height = 48
        max_chunk_size = 16
        threshold = 0.2 if config.prob is None else config.prob

        quadrilaterals = list(self._generate_text_direction(textlines))
        region_imgs = [q.get_transformed_region(image, d, text_height) for q, d in quadrilaterals]
        out_regions = []

        perm = range(len(region_imgs))
        is_quadrilaterals = False
        if len(quadrilaterals) > 0 and isinstance(quadrilaterals[0][0], Quadrilateral):
            perm = sorted(range(len(region_imgs)), key = lambda x: region_imgs[x].shape[1])
            is_quadrilaterals = True

        ix = 0
        for indices in chunks(perm, max_chunk_size):
            N = len(indices)
            widths = [region_imgs[i].shape[1] for i in indices]
            max_width = 4 * (max(widths) + 7) // 4
            region = np.zeros((N, text_height, max_width, 3), dtype = np.uint8)
            for i, idx in enumerate(indices):
                W = region_imgs[idx].shape[1]
                tmp = region_imgs[idx]
                region[i, :, : W, :]=tmp
                if verbose:
                    os.makedirs('result/ocrs/', exist_ok=True)
                    if quadrilaterals[idx][1] == 'v':
                        cv2.imwrite(f'result/ocrs/{ix}.png', cv2.rotate(cv2.cvtColor(region[i, :, :, :], cv2.COLOR_RGB2BGR), cv2.ROTATE_90_CLOCKWISE))
                    else:
                        cv2.imwrite(f'result/ocrs/{ix}.png', cv2.cvtColor(region[i, :, :, :], cv2.COLOR_RGB2BGR))
                ix += 1
            image_tensor = (torch.from_numpy(region).float() - 127.5) / 127.5
            image_tensor = einops.rearrange(image_tensor, 'N H W C -> N C H W')
            if self.use_gpu:
                image_tensor = image_tensor.to(self.device)
            with torch.no_grad():
                ret = self.model.infer_beam_batch_tensor(image_tensor, widths, beams_k = 5, max_seq_length = 255)
            for i, (pred_chars_index, prob, fg_pred, bg_pred, fg_ind_pred, bg_ind_pred) in enumerate(ret):
                if prob < threshold:
                    continue
                has_fg = (fg_ind_pred[:, 1] > fg_ind_pred[:, 0])
                has_bg = (bg_ind_pred[:, 1] > bg_ind_pred[:, 0])
                seq = []
                fr = AvgMeter()
                fg = AvgMeter()
                fb = AvgMeter()
                br = AvgMeter()
                bg = AvgMeter()
                bb = AvgMeter()
                for chid, c_fg, c_bg, h_fg, h_bg in zip(pred_chars_index, fg_pred, bg_pred, has_fg, has_bg) :
                    ch = self.model.dictionary[chid]
                    if ch == '<S>':
                        continue
                    if ch == '</S>':
                        break
                    if ch == '<SP>':
                        ch = ' '
                    seq.append(ch)
                    if h_fg.item() :
                        fr(int(c_fg[0] * 255))
                        fg(int(c_fg[1] * 255))
                        fb(int(c_fg[2] * 255))
                    if h_bg.item() :
                        br(int(c_bg[0] * 255))
                        bg(int(c_bg[1] * 255))
                        bb(int(c_bg[2] * 255))
                    else :
                        br(int(c_fg[0] * 255))
                        bg(int(c_fg[1] * 255))
                        bb(int(c_fg[2] * 255))
                txt = ''.join(seq)
                fr = min(max(int(fr()), 0), 255)
                fg = min(max(int(fg()), 0), 255)
                fb = min(max(int(fb()), 0), 255)
                br = min(max(int(br()), 0), 255)
                bg = min(max(int(bg()), 0), 255)
                bb = min(max(int(bb()), 0), 255)
                self.logger.info(f'prob: {prob} {txt} fg: ({fr}, {fg}, {fb}) bg: ({br}, {bg}, {bb})')
                cur_region = quadrilaterals[indices[i]][0]
                if isinstance(cur_region, Quadrilateral):
                    cur_region.text = txt
                    cur_region.prob = prob
                    cur_region.fg_r = fr
                    cur_region.fg_g = fg
                    cur_region.fg_b = fb
                    cur_region.bg_r = br
                    cur_region.bg_g = bg
                    cur_region.bg_b = bb
                else:
                    cur_region.text.append(txt)
                    cur_region.update_font_colors(np.array([fr, fg, fb]), np.array([br, bg, bb]))

                out_regions.append(cur_region)

        if is_quadrilaterals:
            return out_regions
        return textlines

class ConvNeXtBlock(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, layer_scale_init_value=1e-6, ks = 7, padding = 3):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=ks, padding=padding, groups=dim) # depthwise conv
        self.norm = nn.BatchNorm2d(dim, eps=1e-6)
        self.pwconv1 = nn.Conv2d(dim, 4 * dim, 1, 1, 0) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(4 * dim, dim, 1, 1, 0)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(1, dim, 1, 1), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x

        x = input + x
        return x

class ConvNext_FeatureExtractor(nn.Module) :
    def __init__(self, img_height = 48, in_dim = 3, dim = 512, n_layers = 12) -> None:
        super().__init__()
        base = dim // 8
        self.stem = nn.Sequential(
            nn.Conv2d(in_dim, base, kernel_size = 7, stride = 1, padding = 3),
            nn.BatchNorm2d(base),
            nn.ReLU(),
            nn.Conv2d(base, base * 2, kernel_size = 2, stride = 2, padding = 0),
            nn.BatchNorm2d(base * 2),
            nn.ReLU(),
            nn.Conv2d(base * 2, base * 2, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(base * 2),
            nn.ReLU(),
        )
        self.block1 = self.make_layers(base * 2, 4)
        self.down1 = nn.Sequential(
            nn.Conv2d(base * 2, base * 4, kernel_size = 2, stride = 2, padding = 0),
            nn.BatchNorm2d(base * 4),
            nn.ReLU(),
        )
        self.block2 = self.make_layers(base * 4, 12)
        self.down2 = nn.Sequential(
            nn.Conv2d(base * 4, base * 8, kernel_size = (2, 1), stride = (2, 1), padding = (0, 0)),
            nn.BatchNorm2d(base * 8),
            nn.ReLU(),
        )
        self.block3 = self.make_layers(base * 8, 10, ks = 5, padding = 2)
        self.down3 = nn.Sequential(
            nn.Conv2d(base * 8, base * 8, kernel_size = (2, 1), stride = (2, 1), padding = (0, 0)),
            nn.BatchNorm2d(base * 8),
            nn.ReLU(),
        )
        self.block4 = self.make_layers(base * 8, 8, ks = 3, padding = 1)
        self.down4 = nn.Sequential(
            nn.Conv2d(base * 8, base * 8, kernel_size = (3, 1), stride = (1, 1), padding = (0, 0)),
            nn.BatchNorm2d(base * 8),
            nn.ReLU(),
        )

    def make_layers(self, dim, n, ks = 7, padding = 3) :
        layers = []
        for i in range(n) :
            layers.append(ConvNeXtBlock(dim, ks = ks, padding = padding))
        return nn.Sequential(*layers)
    
    def forward(self, x) :
        x = self.stem(x)
        # h//2, w//2
        x = self.block1(x)
        x = self.down1(x)
        # h//4, w//4
        x = self.block2(x)
        x = self.down2(x)
        # h//8, w//4
        x = self.block3(x)
        x = self.down3(x)
        # h//16, w//4
        x = self.block4(x)
        x = self.down4(x)
        return x

def transformer_encoder_forward(
    self,
    src: torch.Tensor,
    src_mask: Optional[torch.Tensor] = None,
    src_key_padding_mask: Optional[torch.Tensor] = None,
    is_causal: bool = False) -> torch.Tensor:
    x = src
    if self.norm_first:
        x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
        x = x + self._ff_block(self.norm2(x))
    else:
        x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
        x = self.norm2(x + self._ff_block(x))

    return x

class XposMultiheadAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        self_attention=False,
        encoder_decoder_attention=False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim**-0.5

        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention
        assert self.self_attention ^ self.encoder_decoder_attention

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias = True)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias = True)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias = True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias = True)
        self.xpos = XPOS(self.head_dim, embed_dim)
        self.batch_first = True
        self._qkv_same_embed_dim = True

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(
        self,
        query,
        key,
        value,
        key_padding_mask=None,
        attn_mask=None,
        need_weights = False,
        is_causal = False,
        k_offset = 0,
        q_offset = 0
    ):
        assert not is_causal
        bsz, tgt_len, embed_dim = query.size()
        src_len = tgt_len
        assert embed_dim == self.embed_dim, f"query dim {embed_dim} != {self.embed_dim}"

        key_bsz, src_len, _ = key.size()
        assert key_bsz == bsz, f"{query.size(), key.size()}"
        assert value is not None
        assert bsz, src_len == value.shape[:2]

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        q *= self.scaling

        q = q.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        q = q.reshape(bsz * self.num_heads, tgt_len, self.head_dim)
        k = k.reshape(bsz * self.num_heads, src_len, self.head_dim)
        v = v.reshape(bsz * self.num_heads, src_len, self.head_dim)

        if self.xpos is not None:
            k = self.xpos(k, offset=k_offset, downscale=True) # TODO: read paper
            q = self.xpos(q, offset=q_offset, downscale=False)

        attn_weights = torch.bmm(q, k.transpose(1, 2))

        if attn_mask is not None:
            attn_weights = torch.nan_to_num(attn_weights)
            attn_mask = attn_mask.unsqueeze(0)
            attn_weights += attn_mask

        if key_padding_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                float("-inf"),
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(
            attn_weights
        )
        attn = torch.bmm(attn_weights, v)
        attn = attn.transpose(0, 1).reshape(tgt_len, bsz, embed_dim).transpose(0, 1)

        attn = self.out_proj(attn)
        attn_weights = attn_weights.view(
            bsz, self.num_heads, tgt_len, src_len
        ).transpose(1, 0)

        if need_weights:
            return attn, attn_weights
        else :
            return attn, None

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

class Beam:
    def __init__(self, char_seq = [], logprobs = []):
        # L
        if isinstance(char_seq, list):
            self.chars = torch.tensor(char_seq, dtype=torch.long)
            self.logprobs = torch.tensor(logprobs, dtype=torch.float32)
        else:
            self.chars = char_seq.clone()
            self.logprobs = logprobs.clone()

    def avg_logprob(self):
        return self.logprobs.mean().item()

    def sort_key(self):
        return -self.avg_logprob()

    def seq_end(self, end_tok):
        return self.chars.view(-1)[-1] == end_tok

    def extend(self, idx, logprob):
        return Beam(
            torch.cat([self.chars, idx.unsqueeze(0)], dim = -1),
            torch.cat([self.logprobs, logprob.unsqueeze(0)], dim = -1),
        )

DECODE_BLOCK_LENGTH = 8

class Hypothesis:
    def __init__(self, device, start_tok: int, end_tok: int, padding_tok: int, memory_idx: int, num_layers: int, embd_dim: int):
        self.device = device
        self.start_tok = start_tok
        self.end_tok = end_tok
        self.padding_tok = padding_tok
        self.memory_idx = memory_idx
        self.embd_size = embd_dim
        self.num_layers = num_layers
        # 1, L, E
        self.cached_activations = [torch.zeros(1, 0, self.embd_size).to(self.device)] * (num_layers + 1)
        self.out_idx = torch.LongTensor([start_tok]).to(self.device)
        self.out_logprobs = torch.FloatTensor([0]).to(self.device)
        self.length = 0

    def seq_end(self):
        return self.out_idx.view(-1)[-1] == self.end_tok

    def logprob(self):
        return self.out_logprobs.mean().item()

    def sort_key(self):
        return -self.logprob()

    def prob(self):
        return self.out_logprobs.mean().exp().item()

    def __len__(self):
        return self.length

    def extend(self, idx, logprob):
        ret = Hypothesis(self.device, self.start_tok, self.end_tok, self.padding_tok, self.memory_idx, self.num_layers, self.embd_size)
        ret.cached_activations = [item.clone() for item in self.cached_activations]
        ret.length = self.length + 1
        ret.out_idx = torch.cat([self.out_idx, torch.LongTensor([idx]).to(self.device)], dim = 0)
        ret.out_logprobs = torch.cat([self.out_logprobs, torch.FloatTensor([logprob]).to(self.device)], dim = 0)
        return ret

    def output(self):
        return self.cached_activations[-1]

def next_token_batch(
    hyps: List[Hypothesis],
    memory: torch.Tensor, # N, H, W, C
    memory_mask: torch.BoolTensor,
    decoders: nn.ModuleList,
    embd: nn.Embedding
    ):
    layer: nn.TransformerDecoderLayer
    N = len(hyps)
    offset = len(hyps[0])

    # N
    last_toks = torch.stack([item.out_idx[-1] for item in hyps])
    # N, 1, E
    tgt: torch.FloatTensor = embd(last_toks).unsqueeze_(1)
    
    # N, L, E
    memory = torch.stack([memory[idx, :, :] for idx in [item.memory_idx for item in hyps]], dim = 0)
    for l, layer in enumerate(decoders):
        # TODO: keys and values are recomputed every time
        # N, L - 1, E
        combined_activations = torch.cat([item.cached_activations[l] for item in hyps], dim = 0)
        # N, L, E
        combined_activations = torch.cat([combined_activations, tgt], dim = 1)
        for i in range(N):
            hyps[i].cached_activations[l] = combined_activations[i: i + 1, :, :]
        # N, 1, E
        tgt = tgt + layer.self_attn(layer.norm1(tgt), layer.norm1(combined_activations), layer.norm1(combined_activations), q_offset = offset)[0]
        tgt = tgt + layer.multihead_attn(layer.norm2(tgt), memory, memory, key_padding_mask = memory_mask, q_offset = offset)[0]
        tgt = tgt + layer._ff_block(layer.norm3(tgt))
    #print(tgt[0, 0, 0])
    for i in range(N):
        hyps[i].cached_activations[len(decoders)] = torch.cat([hyps[i].cached_activations[len(decoders)], tgt[i: i + 1, :, :]], dim = 1)
    # N, E
    return tgt.squeeze_(1)

class OCR(nn.Module):
    def __init__(self, dictionary, max_len):
        super(OCR, self).__init__()
        self.max_len = max_len
        self.dictionary = dictionary
        self.dict_size = len(dictionary)
        n_decoders = 4
        embd_dim = 320
        nhead = 4
        #self.backbone = LocalViT_FeatureExtractor(48, 3, dim = embd_dim, ff_dim = embd_dim * 4, n_layers = n_encoders)
        self.backbone = ConvNext_FeatureExtractor(48, 3, embd_dim)
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        
        for i in range(4) :
            encoder = nn.TransformerEncoderLayer(embd_dim, nhead, dropout = 0, batch_first = True, norm_first = True)
            encoder.self_attn = XposMultiheadAttention(embd_dim, nhead, self_attention = True)
            encoder.forward = transformer_encoder_forward
            self.encoders.append(encoder)
        self.encoders.forward = self.encoder_forward
        
        for i in range(5) :
            decoder = nn.TransformerDecoderLayer(embd_dim, nhead, dropout = 0, batch_first = True, norm_first = True)
            decoder.self_attn = XposMultiheadAttention(embd_dim, nhead, self_attention = True)
            decoder.multihead_attn = XposMultiheadAttention(embd_dim, nhead, encoder_decoder_attention = True)
            self.decoders.append(decoder)
        self.decoders.forward = self.decoder_forward
        
        self.embd = nn.Embedding(self.dict_size, embd_dim)
        self.pred1 = nn.Sequential(nn.Linear(embd_dim, embd_dim), nn.GELU(), nn.Dropout(0.15))
        self.pred = nn.Linear(embd_dim, self.dict_size)
        self.pred.weight = self.embd.weight
        self.color_pred1 = nn.Sequential(nn.Linear(embd_dim, 64), nn.ReLU())
        self.color_pred_fg = nn.Linear(64, 3)
        self.color_pred_bg = nn.Linear(64, 3)
        self.color_pred_fg_ind = nn.Linear(64, 2)
        self.color_pred_bg_ind = nn.Linear(64, 2)

    def encoder_forward(self, memory, encoder_mask):
        for layer in self.encoders :
            memory = layer(layer, src = memory, src_key_padding_mask = encoder_mask)
        return memory

    def decoder_forward(
        self,
        embd: torch.Tensor,
        cached_activations: torch.Tensor,  # Shape [N, L, T, E] where L=num_layers, T=sequence length, E=embedding size
        memory: torch.Tensor,  # Shape [N, H, W, C] (Encoder memory output)
        memory_mask: torch.BoolTensor,
        step: int
    ):

        layer: nn.TransformerDecoderLayer
        tgt = embd  # N, 1, E for the last token embedding

        for l, layer in enumerate(self.decoders):
            combined_activations = cached_activations[:, l, :step, :]  # N, T, E
            combined_activations = torch.cat([combined_activations, tgt], dim=1)  # N, T+1, E
            cached_activations[:, l, step, :] = tgt.squeeze(1)

            # Update cache and perform self attention
            tgt = tgt + layer.self_attn(layer.norm1(tgt), layer.norm1(combined_activations), layer.norm1(combined_activations), q_offset=step)[0]
            tgt = tgt + layer.multihead_attn(layer.norm2(tgt), memory, memory, key_padding_mask=memory_mask, q_offset=step)[0]
            tgt = tgt + layer._ff_block(layer.norm3(tgt))

        cached_activations[:, l+1, step, :] = tgt.squeeze(1) # Append the new activations

        return tgt.squeeze_(1), cached_activations

    def forward(self,
        img: torch.FloatTensor,
        char_idx: torch.LongTensor,
        decoder_mask: torch.BoolTensor,
        encoder_mask: torch.BoolTensor
        ):
        memory = self.backbone(img)
        memory = einops.rearrange(memory, 'N C 1 W -> N W C')
        for layer in self.encoders :
            memory = layer(memory, src_key_padding_mask = encoder_mask)
        N, L = char_idx.shape
        char_embd = self.embd(char_idx)
        # N, L, D
        casual_mask = generate_square_subsequent_mask(L).to(img.device)
        decoded = char_embd
        for layer in self.decoders :
            decoded = layer(decoded, memory, tgt_mask = casual_mask, tgt_key_padding_mask = decoder_mask, memory_key_padding_mask = encoder_mask)
        
        pred_char_logits = self.pred(self.pred1(decoded))
        color_feats = self.color_pred1(decoded)
        return pred_char_logits, \
            self.color_pred_fg(color_feats), \
            self.color_pred_bg(color_feats), \
            self.color_pred_fg_ind(color_feats), \
            self.color_pred_bg_ind(color_feats)

    def infer_beam_batch(self, img: torch.FloatTensor, img_widths: List[int], beams_k: int = 5, start_tok = 1, end_tok = 2, pad_tok = 0, max_finished_hypos: int = 2, max_seq_length = 384):
        N, C, H, W = img.shape
        assert H == 48 and C == 3
        memory = self.backbone(img)
        memory = einops.rearrange(memory, 'N C 1 W -> N W C')
        valid_feats_length = [(x + 3) // 4 + 2 for x in img_widths]
        input_mask = torch.zeros(N, memory.size(1), dtype = torch.bool).to(img.device)
        for i, l in enumerate(valid_feats_length):
            input_mask[i, l:] = True
        for layer in self.encoders :
            memory = layer(layer, src = memory, src_key_padding_mask = input_mask)
        hypos = [Hypothesis(img.device, start_tok, end_tok, pad_tok, i, len(self.decoders), 320) for i in range(N)]
        # N, E
        decoded = next_token_batch(hypos, memory, input_mask, self.decoders, self.embd)
        # N, n_chars
        pred_char_logprob = self.pred(self.pred1(decoded)).log_softmax(-1)
        # N, k
        pred_chars_values, pred_chars_index = torch.topk(pred_char_logprob, beams_k, dim = 1)
        new_hypos: List[Hypothesis] = []
        finished_hypos = defaultdict(list)
        for i in range(N):
            for k in range(beams_k):
                new_hypos.append(hypos[i].extend(pred_chars_index[i, k], pred_chars_values[i, k]))
        hypos = new_hypos
        for ixx in range(max_seq_length):
            # N * k, E
            decoded = next_token_batch(hypos, memory, torch.stack([input_mask[hyp.memory_idx] for hyp in hypos]) , self.decoders, self.embd)
            # N * k, n_chars
            pred_char_logprob = self.pred(self.pred1(decoded)).log_softmax(-1)
            # N * k, k
            pred_chars_values, pred_chars_index = torch.topk(pred_char_logprob, beams_k, dim = 1)
            hypos_per_sample = defaultdict(list)
            h: Hypothesis
            for i, h in enumerate(hypos):
                for k in range(beams_k):
                    hypos_per_sample[h.memory_idx].append(h.extend(pred_chars_index[i, k], pred_chars_values[i, k]))
            hypos = []
            # hypos_per_sample now contains N * k^2 hypos
            for i in hypos_per_sample.keys():
                cur_hypos: List[Hypothesis] = hypos_per_sample[i]
                cur_hypos = sorted(cur_hypos, key = lambda a: a.sort_key())[: beams_k + 1]
                #print(cur_hypos[0].out_idx[-1])
                to_added_hypos = []
                sample_done = False
                for h in cur_hypos:
                    if h.seq_end():
                        finished_hypos[i].append(h)
                        if len(finished_hypos[i]) >= max_finished_hypos:
                            sample_done = True
                            break
                    else:
                        if len(to_added_hypos) < beams_k:
                            to_added_hypos.append(h)
                if not sample_done:
                    hypos.extend(to_added_hypos)
            if len(hypos) == 0:
                break
        # add remaining hypos to finished
        for i in range(N):
            if i not in finished_hypos:
                cur_hypos: List[Hypothesis] = hypos_per_sample[i]
                cur_hypo = sorted(cur_hypos, key = lambda a: a.sort_key())[0]
                finished_hypos[i].append(cur_hypo)
        assert len(finished_hypos) == N
        result = []
        for i in range(N):
            cur_hypos = finished_hypos[i]
            cur_hypo = sorted(cur_hypos, key = lambda a: a.sort_key())[0]
            decoded = cur_hypo.output()
            color_feats = self.color_pred1(decoded)
            fg_pred, bg_pred, fg_ind_pred, bg_ind_pred = \
                self.color_pred_fg(color_feats), \
                self.color_pred_bg(color_feats), \
                self.color_pred_fg_ind(color_feats), \
                self.color_pred_bg_ind(color_feats)
            result.append((cur_hypo.out_idx[1:], cur_hypo.prob(), fg_pred[0], bg_pred[0], fg_ind_pred[0], bg_ind_pred[0]))
        return result

    def infer_beam_batch_tensor(self, img: torch.FloatTensor, img_widths: List[int], beams_k: int = 5, start_tok = 1, end_tok = 2, pad_tok = 0, max_finished_hypos: int = 2, max_seq_length = 384):
        N, C, H, W = img.shape
        assert H == 48 and C == 3

        memory = self.backbone(img)
        memory = einops.rearrange(memory, 'N C 1 W -> N W C')
        valid_feats_length = [(x + 3) // 4 + 2 for x in img_widths]
        input_mask = torch.zeros(N, memory.size(1), dtype = torch.bool).to(img.device)
        
        for i, l in enumerate(valid_feats_length):
            input_mask[i, l:] = True
        memory = self.encoders(memory, input_mask) # N, W, Dim

        out_idx = torch.full((N, 1), start_tok, dtype=torch.long, device=img.device)  # Shape [N, 1]
        cached_activations = torch.zeros(N, len(self.decoders)+1, max_seq_length, 320, device=img.device)  # [N, L, S, E]
        log_probs = torch.zeros(N, 1, device=img.device)  # Shape [N, 1]        # N, E
        idx_embedded = self.embd(out_idx[:, -1:]) 

        decoded, cached_activations = self.decoders(idx_embedded, cached_activations, memory, input_mask, 0)
        pred_char_logprob = self.pred(self.pred1(decoded)).log_softmax(-1)   # N, n_chars
        pred_chars_values, pred_chars_index = torch.topk(pred_char_logprob, beams_k, dim = 1)  # N, k

        out_idx = torch.cat([out_idx.unsqueeze(1).expand(-1, beams_k, -1), pred_chars_index.unsqueeze(-1)], dim=-1).reshape(-1, 2)  # Shape [N * k, 2]
        log_probs = pred_chars_values.view(-1, 1)  # Shape [N * k, 1]
        memory = memory.repeat_interleave(beams_k, dim=0)
        input_mask = input_mask.repeat_interleave(beams_k, dim=0)
        cached_activations = cached_activations.repeat_interleave(beams_k, dim=0)
        batch_index = torch.arange(N).repeat_interleave(beams_k, dim=0).to(img.device)

        finished_hypos = defaultdict(list)
        N_remaining = N

        for step in range(1, max_seq_length):
            idx_embedded = self.embd(out_idx[:, -1:])
            decoded, cached_activations = self.decoders(idx_embedded, cached_activations, memory, input_mask, step)
            pred_char_logprob = self.pred(self.pred1(decoded)).log_softmax(-1)  # Shape [N * k, dict_size]
            pred_chars_values, pred_chars_index = torch.topk(pred_char_logprob, beams_k, dim=1)  # [N * k, k]

            finished = out_idx[:, -1] == end_tok
            pred_chars_values[finished] = 0
            pred_chars_index[finished] = end_tok

            # Extend hypotheses
            new_out_idx = out_idx.unsqueeze(1).expand(-1, beams_k, -1)  # Shape [N * k, k, seq_len]
            new_out_idx = torch.cat([new_out_idx, pred_chars_index.unsqueeze(-1)], dim=-1)  # Shape [N * k, k, seq_len + 1]
            new_out_idx = new_out_idx.view(-1, step + 2)  # Reshape to [N * k^2, seq_len + 1]
            new_log_probs = log_probs.unsqueeze(1).expand(-1, beams_k, -1) + pred_chars_values.unsqueeze(-1)  # Shape [N * k^2, 1]
            new_log_probs = new_log_probs.view(-1, 1)  # [N * k^2, 1]

            # Sort and select top-k hypotheses per sample
            new_out_idx = new_out_idx.view(N_remaining, -1, step + 2)  # [N, k^2, seq_len + 1]
            new_log_probs = new_log_probs.view(N_remaining, -1)  # [N, k^2]
            batch_topk_log_probs, batch_topk_indices = new_log_probs.topk(beams_k, dim=1)  # [N, k]
            
            # Gather the top-k hypotheses based on log probabilities
            expanded_topk_indices = batch_topk_indices.unsqueeze(-1).expand(-1, -1, new_out_idx.shape[-1])  # Shape [N, k, seq_len + 1]
            out_idx = torch.gather(new_out_idx, 1, expanded_topk_indices).reshape(-1, step + 2)  # [N * k, seq_len + 1]
            log_probs = batch_topk_log_probs.view(-1, 1)  # Reshape to [N * k, 1]

            # Check for finished sequences
            finished = (out_idx[:, -1] == end_tok)  # Check if the last token is the end token
            finished = finished.view(N_remaining, beams_k)  # Reshape to [N, k]
            finished_counts = finished.sum(dim=1)  # Count the number of finished hypotheses per sample
            finished_batch_indices = (finished_counts >= max_finished_hypos).nonzero(as_tuple=False).squeeze()

            if finished_batch_indices.numel() == 0:
                continue

            if finished_batch_indices.dim() == 0:
                finished_batch_indices = finished_batch_indices.unsqueeze(0)
            
            for idx in finished_batch_indices:
                batch_log_probs = batch_topk_log_probs[idx]
                best_beam_idx = batch_log_probs.argmax()
                finished_hypos[batch_index[beams_k * idx].item()] = \
                    out_idx[idx * beams_k + best_beam_idx], \
                    torch.exp(batch_log_probs[best_beam_idx]).item(), \
                    cached_activations[idx * beams_k + best_beam_idx] 

            remaining_indexs = []
            for i in range(N_remaining):
                if i not in finished_batch_indices:
                    for j in range(beams_k):
                        remaining_indexs.append(i * beams_k + j)

            if not remaining_indexs:
                break

            N_remaining = int(len(remaining_indexs) / beams_k)
            out_idx = out_idx.index_select(0, torch.tensor(remaining_indexs, device=img.device))
            log_probs = log_probs.index_select(0, torch.tensor(remaining_indexs, device=img.device))
            memory = memory.index_select(0, torch.tensor(remaining_indexs, device=img.device))
            cached_activations = cached_activations.index_select(0, torch.tensor(remaining_indexs, device=img.device))
            input_mask = input_mask.index_select(0, torch.tensor(remaining_indexs, device=img.device))
            batch_index = batch_index.index_select(0, torch.tensor(remaining_indexs, device=img.device))

        # Ensure we have the correct number of finished hypotheses for each sample
        if len(finished_hypos) < N: # Fallback if not enough finished hypos
            for i in range(N):
                if i not in finished_hypos:
                    # Select the best hypothesis available at the end of beam search
                    sample_indices = (batch_index == i).nonzero(as_tuple=True)[0]
                    if sample_indices.numel() > 0:
                        best_hypo_index = sample_indices[0] # Take the first one as fallback
                        finished_hypos[i] = out_idx[best_hypo_index], torch.exp(log_probs[best_hypo_index]).item(), cached_activations[best_hypo_index]
                    else:
                        # If no hypothesis is available at all (very unlikely, but for robustness)
                        finished_hypos[i] = (torch.tensor([end_tok], device=img.device), 0.0, torch.zeros(cached_activations.shape[1:], device=img.device)) # Dummy hypo

        assert len(finished_hypos) == N

        # Final output processing and color predictions
        result = []
        for i in range(N):
            final_idx, prob, decoded = finished_hypos[i] 
            color_feats = self.color_pred1(decoded[-1].unsqueeze(0))
            fg_pred, bg_pred, fg_ind_pred, bg_ind_pred = \
                self.color_pred_fg(color_feats), \
                self.color_pred_bg(color_feats), \
                self.color_pred_fg_ind(color_feats), \
                self.color_pred_bg_ind(color_feats)
            result.append((final_idx[1:], prob, fg_pred[0], bg_pred[0], fg_ind_pred[0], bg_ind_pred[0]))

        return result

import numpy as np

def convert_pl_model(filename: str) :
    sd = torch.load(filename, map_location = 'cpu')['state_dict']
    sd2 = {}
    for k, v in sd.items() :
        k: str
        k = k.removeprefix('model.')
        sd2[k] = v
    return sd2

def test_LocalViT_FeatureExtractor() :
    net = ConvNext_FeatureExtractor(48, 3, 320)
    inp = torch.randn(2, 3, 48, 512)
    out = net(inp)
    print(out.shape)

def test_infer() :
    with open('alphabet-all-v7.txt', 'r') as fp :
        dictionary = [s[:-1] for s in fp.readlines()]
    model = OCR(dictionary, 32)
    model.eval()
    sd = convert_pl_model('epoch=0-step=13000.ckpt')
    model.load_state_dict(sd)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(params)

    img = cv2.cvtColor(cv2.imread('test3.png'), cv2.COLOR_BGR2RGB)
    ratio = img.shape[1] / float(img.shape[0])
    new_w = int(round(ratio * 48))
    #print(img.shape)
    img = cv2.resize(img, (new_w, 48), interpolation=cv2.INTER_AREA)

    img_torch = einops.rearrange((torch.from_numpy(img) / 127.5 - 1.0), 'h w c -> 1 c h w')

    with torch.no_grad() :
        idx, prob, fg_pred, bg_pred, fg_ind_pred, bg_ind_pred = model.infer_beam_batch(img_torch, [new_w], 5, max_seq_length = 32)[0]
        txt = ''
        for i in idx :
            txt += dictionary[i]
        print(txt, prob)
        for chid, fg, bg, fg_ind, bg_ind in zip(idx, fg_pred[0], bg_pred[0], fg_ind_pred[0], bg_ind_pred[0]) :
            has_fg = (fg_ind[1] > fg_ind[0]).item()
            has_bg = (bg_ind[1] > bg_ind[0]).item()
            if has_fg :
                fg = np.clip((fg * 255).numpy(), 0, 255)
            if has_bg :
                bg = np.clip((bg * 255).numpy(), 0, 255)
            print(f'{dictionary[chid]} {fg if has_fg else "None"} {bg if has_bg else "None"}')

if __name__ == "__main__":
    test_infer()
