# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import einops
import numpy as np
import torch
import torch.nn as nn

def fixed_pos_embedding(x):
    seq_len, dim = x.shape
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim) / dim))
    sinusoid_inp = (
        torch.einsum("i , j -> i j", torch.arange(0, seq_len, dtype=torch.float), inv_freq).to(x)
    )
    return torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)

def rotate_every_two(x):
    x1 = x[:, :, ::2]
    x2 = x[:, :, 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')\

def duplicate_interleave(m):
    """
    A simple version of `torch.repeat_interleave` for duplicating a matrix while interleaving the copy.
    """
    dim0 = m.shape[0]
    m = m.view(-1, 1)  # flatten the matrix
    m = m.repeat(1, 2)  # repeat all elements into the 2nd dimension
    m = m.view(dim0, -1)  # reshape into a matrix, interleaving the copy
    return m

def apply_rotary_pos_emb(x, sin, cos, scale=1):
    sin, cos = map(lambda t: duplicate_interleave(t * scale), (sin, cos))
    # einsum notation for lambda t: repeat(t[offset:x.shape[1]+offset,:], "n d -> () n () (d j)", j=2)
    return (x * cos) + (rotate_every_two(x) * sin)

def apply_rotary_pos_emb2d(x, sin, cos, scale=1):
    breakpoint()
    sin, cos = map(lambda t: duplicate_interleave(t * scale), (sin, cos))
    # einsum notation for lambda t: repeat(t[offset:x.shape[1]+offset,:], "n d -> () n () (d j)", j=2)
    return (x * cos) + (rotate_every_two(x) * sin)

class XPOS(nn.Module):
    def __init__(
        self, head_dim, scale_base=512
    ):
        super().__init__()
        self.head_dim = head_dim
        self.scale_base = scale_base
        self.register_buffer(
            "scale", (torch.arange(0, head_dim, 2) + 0.4 * head_dim) / (1.4 * head_dim)
        )

    def forward(self, x, offset=0, downscale=False):
        length = x.shape[1]
        min_pos = -(length + offset) // 2
        max_pos = length + offset + min_pos
        scale = self.scale ** torch.arange(min_pos, max_pos, 1).to(self.scale).div(self.scale_base)[:, None]
        sin, cos = fixed_pos_embedding(scale)

        if scale.shape[0] > length:
            scale = scale[-length:]
            sin = sin[-length:]
            cos = cos[-length:]
        
        if downscale:
            scale = 1 / scale

        x = apply_rotary_pos_emb(x, sin, cos, scale)
        return x


class XPOS2D(nn.Module):
    def __init__(
        self, head_dim, scale_base=512
    ):
        super().__init__()
        self.xpos = XPOS(head_dim // 2, scale_base)

    def forward(self, x: torch.Tensor, offset_x = 0, offset_y = 0, downscale=False):
        """
            x: N, H, W, C
        """
        N, H, W, C = x.shape
        C = C // 2
        [dir_x, dir_y] = x.chunk(2, dim = 3)
        dir_x = einops.rearrange(dir_x, 'N H W C -> (N H) W C', N = N, H = H, W = W, C = C)
        dir_y = einops.rearrange(dir_y, 'N H W C -> (N W) H C', N = N, H = H, W = W, C = C)
        dir_x = self.xpos(dir_x, offset = offset_x, downscale = downscale)
        dir_y = self.xpos(dir_y, offset = offset_y, downscale = downscale)
        dir_x = einops.rearrange(dir_x, '(N H) W C -> N H W C', N = N, H = H, W = W, C = C)
        dir_y = einops.rearrange(dir_y, '(N W) H C -> N H W C', N = N, H = H, W = W, C = C)
        return torch.cat([dir_x, dir_y], dim = 3)

def test() :
    e = XPOS2D(64, 512)
    x = torch.randn(8, 10, 10, 64)
    o = e(x)
    print(o.shape)

if __name__ == '__main__' :
    test()
