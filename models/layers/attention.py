import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
        ndim = x.ndim
        assert 0 <= 1 < ndim
        assert freqs_cis.shape == (x.shape[1], x.shape[-1])
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return freqs_cis.view(*shape)

    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )

class MultiheadAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        num_kv_heads=None,
        dropout=0.0,
        bias=True,
        self_attention=False,
        q_noise=0.0,
        qn_block_size=8,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else 8
        self.num_local_heads = num_heads
        self.num_local_kv_heads = self.num_kv_heads
        self.num_rep = self.num_local_heads // self.num_local_kv_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5

        self.wq = nn.Linear(embed_dim, num_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(embed_dim, self.num_kv_heads * self.head_dim, bias=bias)
        self.wv = nn.Linear(embed_dim, self.num_kv_heads * self.head_dim, bias=bias)
        self.wo = nn.Linear(num_heads * self.head_dim, embed_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x,
        attn_bias=None,
        key_padding_mask=None,
        need_weights=True,
        attn_mask=None,
        freqs_cis=None,
        start_pos=0,
    ):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.num_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.num_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.num_local_kv_heads, self.head_dim)

        if freqs_cis is not None:
            xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        keys = xk
        values = xv

        keys = repeat_kv(keys, self.num_rep)
        values = repeat_kv(values, self.num_rep)

        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # QK^T / sqrt(d)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)

        # [Fix] 强制 Clamp，防止 Transformer 早期训练不稳定导致 Score 爆炸
        scores = torch.clamp(scores, min=-50000, max=50000)

        if attn_bias is not None:
            scores += attn_bias

        if attn_mask is not None:
            scores = scores + attn_mask.unsqueeze(1)

        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        # [Fix] 强制使用 float32 做 softmax，避免 fp16 下的溢出
        attn_weights = F.softmax(scores.float(), dim=-1).type_as(xq)
        attn_weights = self.dropout(attn_weights)

        output = torch.matmul(attn_weights, values)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        output = self.wo(output)

        return output, attn_weights