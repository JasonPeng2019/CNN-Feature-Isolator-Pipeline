"""SAE families A (expressive conv field) and B (shared vector / patchwise).

Both map a hidden field H_t (B,C,H,W) -> coefficient field Z (B,K,H,W) -> Hhat.
The mask (masks.py) is applied to Z between encode() and decode(). There is NO
input->output skip around Z (forbidden by the design); internal residuals inside
LocalResBlocks are allowed.

Decoder weights are L2-normalised on the feature axis (standard SAE practice) for
Type B; for Type A the final 1x1 conv is normalised per output unit.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalResBlock(nn.Module):
    """ConvNeXt-style: DWConv(k) -> Norm -> 1x1 expand -> GELU -> 1x1 proj -> +x."""
    def __init__(self, d, k=5, expansion=4):
        super().__init__()
        self.dw = nn.Conv2d(d, d, k, 1, k // 2, groups=d)
        self.norm = nn.GroupNorm(1, d)  # LayerNorm over channels, spatial-preserving
        self.pw1 = nn.Conv2d(d, expansion * d, 1)
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(expansion * d, d, 1)

    def forward(self, x):
        return x + self.pw2(self.act(self.pw1(self.norm(self.dw(x)))))


class FieldSAE(nn.Module):
    """Type A: expressive convolutional field SAE."""
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
        return F.relu(self.enc_out(x))  # nonneg coefficients

    def decode(self, z):
        x = F.gelu(self.dec_in(z))
        x = self.dec_blocks(x)
        return self.dec_out(x)


class VectorSAE(nn.Module):
    """Type B: shared vector SAE applied at every location (1x1 convs == per-location linear).
    Standard tied-free SAE with normalised decoder columns and a learned pre-bias."""
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
        w = self.dec.weight  # (C,K,1,1)
        self.dec.weight.copy_(w / (w.norm(dim=0, keepdim=True) + 1e-8))

    def encode(self, h):
        return F.relu(self.enc(h - self.pre_bias))

    def decode(self, z):
        return self.dec(z) + self.dec_bias


def build_sae(sae_type, C, K, d=None, n_blocks=3, k=5):
    if sae_type == "field":
        return FieldSAE(C, K, d=d, n_blocks=n_blocks, k=k)
    if sae_type == "vector":
        return VectorSAE(C, K)
    raise ValueError(sae_type)
