"""
AMViT -- Adaptive Memory Vision Transformer
=============================================

An implementation of the Adaptive Memory Mechanism (AMM) described in

    "Adaptive Memory Mechanism in Vision Transformer for
     Long-Form Video Understanding"          (ICLR 2025 submission, id=1DEHVMDBaO)

AMM extends memory-augmented ViTs (e.g. MeMViT / Transformer-XL-style KV
caching) with a *Memory Bank*: instead of throwing away the oldest KV chunk
once the fixed memory length `m` is exceeded (plain FIFO), the tokens most
relevant to the current Class Token (CLS) query are folded into a persistent
Memory Bank before the chunk is discarded, so useful long-range information
can survive arbitrarily far back in the video.

Written the way Andrej Karpathy writes nanoGPT / GPT-2: one file, a
@dataclass config, small explicit nn.Module building blocks
(LayerNorm -> MLP -> Attention -> Block -> full model), manual tensor
surgery instead of einops, and a __main__ smoke test at the bottom.

Reference equations (paper section 3.3, appendix Algorithms 1 & 2):

    selection(K, q0, k)   = top-k keys/values by <k_j, q0>          (Alg. 1, Eq. 7)
    MB_i  = [ selection(cpr(K_{i-m-1}), q0, L-k'), selection(MB_{i-1}, q0, k') ]   (Eq. 8, 9)
    K_bar_i = [ MB_i, selection(cpr(K_{i-m}), q0, k''), ..., selection(cpr(K_{i-1}), q0, k''), K_i ]  (Eq. 10, 11)
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

@dataclass
class AMViTConfig:
    # input / patchify
    img_size: int = 224
    patch_size: int = 16
    in_chans: int = 3
    frames_per_segment: int = 8          # t: frames processed per forward() call

    # transformer
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True

    # Adaptive Memory Mechanism
    memory_length: int = 2               # m: number of raw KV chunks kept before folding into the bank
    cpr_tokens: int = 192                # size each raw KV chunk is compressed to before caching ("cpr")
    sel_tokens: int = 50                 # k'': tokens selected per cached chunk when building K_bar (Eq. 10/11)
    bank_size: int = 50                  # L: size of the persistent Memory Bank
    bank_ratio: float = 0.2              # alpha: fraction of the bank kept from MB_{i-1} each update

    num_classes: int = 400

    @property
    def n_patches(self) -> int:
        return (self.img_size // self.patch_size) ** 2

    @property
    def tokens_per_segment(self) -> int:
        return self.frames_per_segment * self.n_patches + 1  # +1 for CLS


# -----------------------------------------------------------------------------
# Algorithm 1: top-k token selection w.r.t. the CLS query
# -----------------------------------------------------------------------------

def kv_selection(q_cls: torch.Tensor, cached_k, cached_v, k_len: int):
    """
    Select the `k_len` cached tokens most relevant to the current CLS query.

    q_cls:      (B, H, 1, D) -- CLS token's query, current iteration
    cached_k/v: (B, H, N, D) -- candidate tokens to select from
    returns:    (B, H, k_len, D), (B, H, k_len, D)  or  (None, None)
    """
    if cached_k is None or cached_k.shape[2] == 0 or k_len <= 0:
        return None, None
    k_len = min(k_len, cached_k.shape[2])
    # inner product of the CLS query with every candidate key -> (B, H, N)
    atten_score = (q_cls @ cached_k.transpose(-2, -1)).squeeze(2)
    _, idx = torch.topk(atten_score, k_len, dim=-1)                     # (B, H, k_len)
    idx = idx.unsqueeze(-1).expand(-1, -1, -1, cached_k.size(-1))       # (B, H, k_len, D)
    selected_k = torch.gather(cached_k, 2, idx)
    selected_v = torch.gather(cached_v, 2, idx)
    return selected_k, selected_v


def _cat(parts):
    parts = [p for p in parts if p is not None]
    return torch.cat(parts, dim=2) if parts else None


# -----------------------------------------------------------------------------
# "cpr": the compression layer applied to a stop-gradient KV chunk before it
# is admitted into the cache. The paper uses a Convolution Layer; here we use
# a pointwise conv (channel mixing) followed by adaptive average pooling
# along the token dimension, which compresses an arbitrary-length chunk down
# to a fixed `out_tokens` size, matching Table 6 (# of cpr(kv) = 192).
# -----------------------------------------------------------------------------

class KVCompressor(nn.Module):
    def __init__(self, n_embd: int, out_tokens: int):
        super().__init__()
        self.out_tokens = out_tokens
        self.mix = nn.Conv1d(n_embd, n_embd, kernel_size=1)

    def forward(self, x):
        # x: (B, H, N, D) -> merge heads for the conv, then split back
        B, H, N, D = x.shape
        x = x.permute(0, 1, 3, 2).reshape(B, H * D, N)   # (B, H*D, N)
        x = self.mix(x)
        x = F.adaptive_avg_pool1d(x, self.out_tokens)     # (B, H*D, out_tokens)
        x = x.reshape(B, H, D, self.out_tokens).permute(0, 1, 3, 2)  # (B, H, out_tokens, D)
        return x


# -----------------------------------------------------------------------------
# Building blocks (nanoGPT-style)
# -----------------------------------------------------------------------------

class LayerNorm(nn.Module):
    """LayerNorm with an optional bias, since torch's built-in doesn't support bias=False."""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class MLP(nn.Module):
    def __init__(self, config: AMViTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class AMMAttention(nn.Module):
    """
    Multi-head self-attention over the current segment's tokens, augmented
    with an Adaptive Memory Mechanism KV cache (Memory Bank + selective FIFO).

    This module is *stateful*: it keeps `cached_k/v` (recent compressed KV
    chunks) and `bank_k/v` (the persistent Memory Bank) across successive
    calls to forward(), i.e. across successive video segments. Call
    reset_memory() at the start of each new video / clip.
    """

    def __init__(self, config: AMViTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.mem_len = config.memory_length
        self.sel_tokens = config.sel_tokens
        self.bank_size = config.bank_size
        self.k_prime = int(config.bank_ratio * config.bank_size)   # k' in Eq. 8/9

        self.compress_k = KVCompressor(config.n_embd, config.cpr_tokens)
        self.compress_v = KVCompressor(config.n_embd, config.cpr_tokens)

        self.reset_memory()

    def reset_memory(self):
        """Clear all cached KV chunks and the Memory Bank (call between videos)."""
        self.cached_k = []   # list of (B, H, cpr_tokens, D) stop-gradient tensors, oldest first
        self.cached_v = []
        self.bank_k = None   # (B, H, <=bank_size, D)
        self.bank_v = None

    def _split_heads(self, x, B, T):
        return x.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, H, T, D)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(self.n_head * self.head_dim, dim=2)
        q = self._split_heads(q, B, T)
        k = self._split_heads(k, B, T)
        v = self._split_heads(v, B, T)

        cls_q = q[:, :, :1, :]   # CLS token's query drives all selection (Eq. 7)

        # ---- 1. fold the oldest cached chunk into the Memory Bank, if the
        #         cache has grown past the fixed memory length m (Eq. 8/9) ----
        if len(self.cached_k) > self.mem_len:
            oldest_k, oldest_v = self.cached_k.pop(0), self.cached_v.pop(0)
            from_old_len = self.bank_size - self.k_prime
            sel_old_k, sel_old_v = kv_selection(cls_q, oldest_k, oldest_v, from_old_len)
            sel_bank_k, sel_bank_v = kv_selection(cls_q, self.bank_k, self.bank_v, self.k_prime)
            self.bank_k = _cat([sel_old_k, sel_bank_k])
            self.bank_v = _cat([sel_old_v, sel_bank_v])

        # ---- 2. build the memory-augmented K_bar, V_bar (Eq. 10/11) ----
        k_parts, v_parts = [], []

        if self.bank_k is not None:
            k_parts.append(self.bank_k)
            v_parts.append(self.bank_v)

        for ck, cv in zip(self.cached_k, self.cached_v):          # chunks i-m .. i-1
            sk, sv = kv_selection(cls_q, ck, cv, self.sel_tokens)
            k_parts.append(sk)
            v_parts.append(sv)

        k_parts.append(k)          # current segment's own K_i
        v_parts.append(v)

        k_bar = _cat(k_parts)
        v_bar = _cat(v_parts)

        # ---- 3. standard scaled-dot-product attention over the augmented KV ----
        y = F.scaled_dot_product_attention(
            q, k_bar, v_bar,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))

        # ---- 4. cache the current segment's (stop-gradient, compressed) KV
        #         for future iterations, mirroring Transformer-XL's sg(K),sg(V) ----
        with torch.no_grad():
            ck_new = self.compress_k(k.detach())
            cv_new = self.compress_v(v.detach())
        self.cached_k.append(ck_new)
        self.cached_v.append(cv_new)

        return y


class Block(nn.Module):
    def __init__(self, config: AMViTConfig):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, config.bias)
        self.attn = AMMAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, config.bias)
        self.mlp = MLP(config)

    def reset_memory(self):
        self.attn.reset_memory()

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class PatchEmbed3D(nn.Module):
    """Tubelet embedding: a single frame-wise (i.e. 2D-per-frame) patchify, as
    in ViT/VideoMAE. A (B, T, C, H, W) clip becomes (B, T*n_patches, n_embd)."""

    def __init__(self, config: AMViTConfig):
        super().__init__()
        self.proj = nn.Conv2d(
            config.in_chans, config.n_embd,
            kernel_size=config.patch_size, stride=config.patch_size,
        )

    def forward(self, x):
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        x = x.reshape(B * T, C, H, W)
        x = self.proj(x)                       # (B*T, n_embd, H/p, W/p)
        x = x.flatten(2).transpose(1, 2)       # (B*T, n_patches, n_embd)
        x = x.view(B, T, -1, x.size(-1)).flatten(1, 2)  # (B, T*n_patches, n_embd)
        return x


# -----------------------------------------------------------------------------
# Full model
# -----------------------------------------------------------------------------

class AMViT(nn.Module):
    """
    Vision Transformer with the Adaptive Memory Mechanism.

    Usage: process a long video as a stream of short segments (t frames each).
    Call reset_memory() once at the start of a video, then forward() once per
    segment; each AMMAttention layer keeps its own KV cache / Memory Bank
    across calls, giving the model an adaptive, input-dependent temporal
    receptive field that can extend far beyond `t * (memory_length + 1)`
    frames without the quadratic cost of attending to the full history.
    """

    def __init__(self, config: AMViTConfig):
        super().__init__()
        self.config = config

        self.patch_embed = PatchEmbed3D(config)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.n_embd))
        self.pos_embed = nn.Parameter(torch.zeros(1, config.tokens_per_segment, config.n_embd))
        self.drop = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = LayerNorm(config.n_embd, config.bias)
        self.head = nn.Linear(config.n_embd, config.num_classes, bias=True)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.Conv2d, nn.Conv1d)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def reset_memory(self):
        """Call at video/clip boundaries so memory from one video doesn't
        leak into the next (the paper's masking strategy, section 4.2)."""
        for block in self.blocks:
            block.reset_memory()

    def forward(self, frames):
        """
        frames: (B, T, C, H, W) -- one segment of T = config.frames_per_segment frames.
        returns: logits (B, num_classes) predicted from the CLS token of this segment,
                 conditioned on everything retained in memory from earlier segments.
        """
        B = frames.shape[0]
        x = self.patch_embed(frames)                                   # (B, T*n_patches, n_embd)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)                                  # (B, tokens_per_segment, n_embd)
        x = self.drop(x + self.pos_embed)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        cls_out = x[:, 0]
        return self.head(cls_out)

    def forward_video(self, video, segment_boundaries=None):
        """
        Convenience driver for a full long-form video processed online.

        video: (B, N, C, H, W) -- N frames, N a multiple of frames_per_segment.
        segment_boundaries: optional set/list of segment indices at which a
            *new* video starts within this batch (triggers reset_memory()).
        returns: list of per-segment logits, one per processed segment.
        """
        t = self.config.frames_per_segment
        assert video.shape[1] % t == 0, "N frames must be a multiple of frames_per_segment"
        n_segments = video.shape[1] // t

        self.reset_memory()
        outputs = []
        for i in range(n_segments):
            if segment_boundaries is not None and i in segment_boundaries:
                self.reset_memory()
            segment = video[:, i * t:(i + 1) * t]
            outputs.append(self.forward(segment))
        return outputs

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())


# -----------------------------------------------------------------------------
# Smoke test
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)

    config = AMViTConfig(
        img_size=64, patch_size=16, in_chans=3, frames_per_segment=4,
        n_layer=2, n_head=4, n_embd=128,
        memory_length=2, cpr_tokens=32, sel_tokens=16, bank_size=24, bank_ratio=0.2,
        num_classes=10,
    )
    model = AMViT(config)
    print(f"AMViT param count: {model.get_num_params() / 1e6:.2f}M")

    # a toy "long-form video": 6 segments of 4 frames = 24 frames total
    B, n_segments = 2, 6
    video = torch.randn(B, n_segments * config.frames_per_segment, config.in_chans,
                         config.img_size, config.img_size)

    logits = model.forward_video(video, segment_boundaries={3})  # simulate a video cut at segment 3
    for i, l in enumerate(logits):
        print(f"segment {i}: logits shape {tuple(l.shape)}")

    # sanity check: memory bank should have populated after enough segments
    last_block_attn = model.blocks[-1].attn
    bank_shape = None if last_block_attn.bank_k is None else tuple(last_block_attn.bank_k.shape)
    print(f"final memory bank shape (B, H, tokens, D): {bank_shape}")
    print(f"# raw cached KV chunks still held: {len(last_block_attn.cached_k)}")
