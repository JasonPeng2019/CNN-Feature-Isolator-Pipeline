"""Changed SAE families for isolated experiments.

This module preserves the current Field/Vector SAEs and adds a patchwise vector
variant that operates on local activation patches extracted by Unfold/Fold.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalResBlock(nn.Module):
    def __init__(self, d, k=5, expansion=4):
        super().__init__()
        self.dw = nn.Conv2d(d, d, k, 1, k // 2, groups=d)
        self.norm = nn.GroupNorm(1, d)
        self.pw1 = nn.Conv2d(d, expansion * d, 1)
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(expansion * d, d, 1)

    def forward(self, x):
        return x + self.pw2(self.act(self.pw1(self.norm(self.dw(x)))))


class FieldSAE(nn.Module):
    def __init__(self, C, K, d=None, n_blocks=3, k=5):
        super().__init__()
        d = d or 2 * C
        self.C, self.K = C, K
        self.in_norm = nn.GroupNorm(1, C)
        self.enc_in = nn.Conv2d(C, d, 1)
        self.enc_blocks = nn.Sequential(*[LocalResBlock(d, k) for _ in range(n_blocks)])
        self.enc_out = nn.Conv2d(d, K, 1)
        self.dec_in = nn.Conv2d(K, d, 1)
        self.dec_blocks = nn.Sequential(*[LocalResBlock(d, k) for _ in range(n_blocks)])
        self.dec_out = nn.Conv2d(d, C, 1)

    def encode(self, h):
        x = self.enc_in(self.in_norm(h))
        x = self.enc_blocks(F.gelu(x))
        return F.relu(self.enc_out(x))

    def decode(self, z):
        x = F.gelu(self.dec_in(z))
        x = self.dec_blocks(x)
        return self.dec_out(x)


class VectorSAE(nn.Module):
    def __init__(self, C, K, d=None, n_blocks=0, k=5):
        super().__init__()
        self.C, self.K = C, K
        self.pre_bias = nn.Parameter(torch.zeros(1, C, 1, 1))
        self.enc = nn.Conv2d(C, K, 1)
        self.dec = nn.Conv2d(K, C, 1, bias=False)
        self.dec_bias = nn.Parameter(torch.zeros(1, C, 1, 1))
        self._normalize_decoder()

    @torch.no_grad()
    def _normalize_decoder(self):
        w = self.dec.weight
        self.dec.weight.copy_(w / (w.norm(dim=0, keepdim=True) + 1e-8))

    def encode(self, h):
        return F.relu(self.enc(h - self.pre_bias))

    def decode(self, z):
        return self.dec(z) + self.dec_bias


class PatchVectorSAEChanged(nn.Module):
    """Shared patchwise vector SAE over activation patches.

    - overlap mode: stride=1 (CNN-style shifted local windows)
    - disjoint mode: stride=patch_size
    """

    def __init__(self, C, K, patch_size, patch_mode="overlap", patch_stride=None):
        super().__init__()
        self.C, self.K = C, K
        self.patch_size = int(patch_size)
        self.patch_mode = patch_mode
        if patch_stride is None:
            patch_stride = 1 if patch_mode == "overlap" else self.patch_size
        self.patch_stride = int(patch_stride)
        self.padding = self.patch_size // 2 if patch_mode == "overlap" else 0
        self.pre_bias = nn.Parameter(torch.zeros(1, C, 1, 1))
        self.dec_bias = nn.Parameter(torch.zeros(1, C, 1, 1))
        self.patch_dim = C * self.patch_size * self.patch_size
        self.enc = nn.Linear(self.patch_dim, K)
        self.dec_weight = nn.Parameter(torch.empty(K, self.patch_dim))
        nn.init.kaiming_uniform_(self.enc.weight, a=math.sqrt(5))
        nn.init.zeros_(self.enc.bias)
        nn.init.kaiming_uniform_(self.dec_weight, a=math.sqrt(5))
        self._normalize_decoder()
        self._last_meta = None

    @torch.no_grad()
    def _normalize_decoder(self):
        self.dec_weight.copy_(self.dec_weight / (self.dec_weight.norm(dim=1, keepdim=True) + 1e-8))

    def _patches(self, h):
        unfold = F.unfold(h - self.pre_bias, kernel_size=self.patch_size, stride=self.patch_stride, padding=self.padding)
        B, _, L = unfold.shape
        patches = unfold.transpose(1, 2).reshape(B * L, self.patch_dim)
        H_out = (h.shape[2] + 2 * self.padding - self.patch_size) // self.patch_stride + 1
        W_out = (h.shape[3] + 2 * self.padding - self.patch_size) // self.patch_stride + 1
        return patches, B, L, H_out, W_out, h.shape[2], h.shape[3]

    def encode(self, h):
        patches, B, L, H_out, W_out, H, W = self._patches(h)
        z = F.relu(self.enc(patches)).reshape(B, L, self.K).transpose(1, 2).reshape(B, self.K, H_out, W_out)
        self._last_meta = {"H": H, "W": W, "H_out": H_out, "W_out": W_out}
        return z

    def decode(self, z):
        if self._last_meta is None:
            raise RuntimeError("encode() must be called before decode() for PatchVectorSAEChanged")
        B, K, H_out, W_out = z.shape
        meta = self._last_meta
        if (H_out, W_out) != (meta["H_out"], meta["W_out"]):
            raise RuntimeError("decode() received latent grid incompatible with the most recent encode()")
        L = H_out * W_out
        z_flat = z.reshape(B, K, L).transpose(1, 2).reshape(B * L, K)
        patches = torch.matmul(z_flat, self.dec_weight).reshape(B, L, self.patch_dim).transpose(1, 2)
        out = F.fold(
            patches,
            output_size=(meta["H"], meta["W"]),
            kernel_size=self.patch_size,
            stride=self.patch_stride,
            padding=self.padding,
        )
        ones = torch.ones((B, self.patch_dim, L), device=z.device, dtype=z.dtype)
        norm = F.fold(
            ones,
            output_size=(meta["H"], meta["W"]),
            kernel_size=self.patch_size,
            stride=self.patch_stride,
            padding=self.padding,
        ).clamp_min(1e-6)
        return out / norm + self.dec_bias


def build_sae(sae_type, C, K, d=None, n_blocks=3, k=5, patch_size=None, patch_mode="overlap", patch_stride=None):
    if sae_type == "field":
        return FieldSAE(C, K, d=d, n_blocks=n_blocks, k=k)
    if sae_type == "vector":
        return VectorSAE(C, K)
    if sae_type == "patchvec":
        if patch_size is None:
            raise ValueError("patch_size is required for patchvec SAE")
        return PatchVectorSAEChanged(C, K, patch_size=patch_size, patch_mode=patch_mode, patch_stride=patch_stride)
    raise ValueError(sae_type)
