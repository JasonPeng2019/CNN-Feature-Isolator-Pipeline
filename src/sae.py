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

from masks import select_topm


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
    """Historical expressive convolutional field SAE."""
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

    def decode(self, z, support=None):
        x = F.gelu(self.dec_in(z))
        x = self.dec_blocks(x)
        return self.dec_out(x)


class StableFieldBlock(nn.Module):
    """Mixed-dilation residual block used by every new redo family."""
    def __init__(self, d, dilation=1, k=5, expansion=4):
        super().__init__()
        self.dw = nn.Conv2d(d, d, k, 1, (k // 2) * dilation,
                            dilation=dilation, groups=d)
        self.norm = nn.GroupNorm(1, d)
        self.pw1 = nn.Conv2d(d, expansion * d, 1)
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(expansion * d, d, 1)
        self.layer_scale = nn.Parameter(torch.full((d,), 1e-3))
        nn.init.zeros_(self.pw1.bias)
        nn.init.zeros_(self.pw2.bias)

    def forward(self, x):
        update = self.pw2(self.act(self.pw1(self.norm(self.dw(x)))))
        return x + update * self.layer_scale.view(1, -1, 1, 1)


class PooledAttentionCorrection(nn.Module):
    """At-most-8x8 pre-norm attention residual with a zero output projection."""
    def __init__(self, d, width=128, heads=4):
        super().__init__()
        if width % heads:
            raise ValueError("attention width must be divisible by head count")
        self.width = width
        self.in_proj = nn.Conv2d(d, width, 1)
        self.norm_attn = nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(width, heads, batch_first=True)
        self.norm_mlp = nn.LayerNorm(width)
        self.mlp = nn.Sequential(nn.Linear(width, 2 * width), nn.GELU(), nn.Linear(2 * width, width))
        self.out_proj = nn.Conv2d(width, d, 1)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    @staticmethod
    def _positions(height, width, channels, device, dtype):
        # Fixed 2-D sine/cosine coordinates, repeated only to fill width.
        y = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
        x = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        base = torch.stack((xx, yy, torch.sin(torch.pi * xx), torch.cos(torch.pi * yy)), dim=-1)
        return base.reshape(1, height * width, 4).repeat(1, 1, (channels + 3) // 4)[..., :channels]

    def forward(self, x):
        height, width = x.shape[-2:]
        pooled = F.adaptive_avg_pool2d(x, (min(height, 8), min(width, 8)))
        tokens = self.in_proj(pooled).flatten(2).transpose(1, 2)
        tokens = tokens + self._positions(tokens.shape[1] // pooled.shape[-1], pooled.shape[-1],
                                          self.width, tokens.device, tokens.dtype)
        q = self.norm_attn(tokens)
        attended, _ = self.attn(q, q, q, need_weights=False)
        tokens = tokens + attended
        tokens = tokens + self.mlp(self.norm_mlp(tokens))
        pooled = tokens.transpose(1, 2).reshape(tokens.shape[0], self.width, pooled.shape[-2], pooled.shape[-1])
        return self.out_proj(F.interpolate(pooled, size=(height, width), mode="bilinear", align_corners=False))


class ContextualFieldEncoder(nn.Module):
    """Full-resolution mixed-dilation trunk plus one pooled attention correction."""
    def __init__(self, C, d, dilations):
        super().__init__()
        if len(dilations) < 1:
            raise ValueError("at least one contextual block is required")
        self.in_proj = nn.Conv2d(C, d, 1)
        split = len(dilations) // 2
        self.first = nn.Sequential(*[StableFieldBlock(d, dilation) for dilation in dilations[:split]])
        self.attention = PooledAttentionCorrection(d)
        self.second = nn.Sequential(*[StableFieldBlock(d, dilation) for dilation in dilations[split:]])

    def forward(self, h):
        x = F.gelu(self.in_proj(h))
        x = self.first(x)
        x = x + self.attention(x)
        return self.second(x)


class SparseMessageMixin:
    """Common exact-message API for new redo families."""
    def encode_sparse(self, h, m, temperature=0.1):
        scores, values = self.encode_score_value(h)
        return select_topm(values, scores, m, temperature=temperature)


class StrictAdditiveFieldDecoder(nn.Module):
    """Linear multiscale spatial-stamp decoder (3/5/9/15 branches)."""
    def __init__(self, C, K):
        super().__init__()
        self.C, self.K = C, K
        self.branches = nn.ModuleList()
        for kernel in (3, 5, 9, 15):
            stamp = nn.ConvTranspose2d(K, K, kernel, padding=kernel // 2,
                                       groups=K, bias=False)
            mix = nn.Conv2d(K, C, 1, bias=False)
            self.branches.append(nn.ModuleDict({"stamp": stamp, "mix": mix}))

    def forward(self, z):
        output = torch.zeros(z.shape[0], self.C, z.shape[-2], z.shape[-1],
                             device=z.device, dtype=z.dtype)
        for branch in self.branches:
            output = output + branch["mix"](branch["stamp"](z))
        return output

    @torch.no_grad()
    def effective_atom_norms(self, size=31):
        """Frobenius norms of each feature's composite interior atom."""
        center = size // 2
        impulse = torch.zeros(1, self.K, size, size,
                              device=self.branches[0]["stamp"].weight.device,
                              dtype=self.branches[0]["stamp"].weight.dtype)
        norms = []
        for feature in range(self.K):
            impulse.zero_()
            impulse[0, feature, center, center] = 1
            norms.append(self.forward(impulse).squeeze(0).norm())
        return torch.stack(norms)

    @torch.no_grad()
    def _normalize_decoder(self):
        # Normalize each feature's composite interior impulse response. This
        # is the effective atom, not an individual branch parameter.
        size = 31
        center = size // 2
        impulse = torch.zeros(1, self.K, size, size,
                              device=self.branches[0]["stamp"].weight.device,
                              dtype=self.branches[0]["stamp"].weight.dtype)
        for feature in range(self.K):
            impulse.zero_(); impulse[0, feature, center, center] = 1
            atom = self.forward(impulse).squeeze(0)
            norm = atom.norm().clamp_min(1e-8)
            for branch in self.branches:
                branch["mix"].weight[:, feature].div_(norm)


class FieldOldSAE(FieldSAE, SparseMessageMixin):
    """Historical shallow Field topology with redo score/value heads."""
    def __init__(self, C, K, d=None, n_blocks=3, k=5):
        super().__init__(C, K, d=d, n_blocks=n_blocks, k=k)
        hidden = d or 2 * C
        self.score_head = nn.Conv2d(hidden, K, 1)
        self.value_head = nn.Conv2d(hidden, K, 1)
        self.value_head.load_state_dict(self.enc_out.state_dict())
        self.score_head.load_state_dict(self.enc_out.state_dict())

    def encode_score_value(self, h):
        x = self.enc_in(self.in_norm(h))
        x = self.enc_blocks(F.gelu(x))
        return self.score_head(x), self.value_head(x)


class StrictAdditiveFieldSAE(nn.Module, SparseMessageMixin):
    """Field-A: contextual encoder and strictly additive decoder."""
    def __init__(self, C, K, d=None):
        super().__init__()
        self.C, self.K, self.d = C, K, d or min(2 * C, 256)
        self.encoder = ContextualFieldEncoder(C, self.d, (1, 1, 2, 1, 4, 1))
        self.score_head = nn.Conv2d(self.d, K, 1)
        self.value_head = nn.Conv2d(self.d, K, 1)
        self.decoder = StrictAdditiveFieldDecoder(C, K)

    def encode_score_value(self, h):
        x = self.encoder(h)
        return self.score_head(x), self.value_head(x)

    def encode(self, h):
        """Return the unmasked signed values for diagnostics/strict warmup."""
        return self.encode_score_value(h)[1]

    def decode(self, z, support=None):
        return self.decoder(z)

    def _normalize_decoder(self):
        self.decoder._normalize_decoder()


class PracticalRecoveryFieldSAE(StrictAdditiveFieldSAE):
    """Field-B: strict additive path plus sparse-message-only correction."""
    def __init__(self, C, K, d=None):
        super().__init__(C, K, d=d)
        self.correction_in = nn.Conv2d(2 * K, self.d, 1)
        self.correction_blocks = nn.Sequential(*[
            StableFieldBlock(self.d, dilation) for dilation in (1, 1, 2, 1, 4, 1)
        ])
        self.correction_out = nn.Conv2d(self.d, C, 1)
        nn.init.zeros_(self.correction_out.weight)
        nn.init.zeros_(self.correction_out.bias)

    def decode(self, z, support=None):
        if support is None:
            support = z.ne(0)
        strict = self.decoder(z)
        correction_input = torch.cat((z, support.to(dtype=z.dtype)), dim=1)
        correction = self.correction_out(self.correction_blocks(F.gelu(self.correction_in(correction_input))))
        return strict + correction

    def strict_decode(self, z):
        return self.decoder(z)


class PracticalContextVectorSAE(nn.Module, SparseMessageMixin):
    """Practical Vector family with direct atoms and sparse-code-only refiner."""
    def __init__(self, C, K, d=None):
        super().__init__()
        self.C, self.K, self.d = C, K, d or min(2 * C, 256)
        self.encoder = ContextualFieldEncoder(C, self.d, (1, 2, 4, 1))
        self.score_head = nn.Conv2d(self.d, K, 1)
        self.value_head = nn.Conv2d(self.d, K, 1)
        self.direct = nn.Conv2d(K, C, 1, bias=False)
        self.dec_bias = nn.Parameter(torch.zeros(1, C, 1, 1))
        self.refiner_in = nn.Conv2d(K, self.d, 1)
        self.refiner_blocks = nn.Sequential(*[
            StableFieldBlock(self.d, dilation) for dilation in (1, 2, 4, 1)
        ])
        self.refiner_out = nn.Conv2d(self.d, C, 1)
        nn.init.zeros_(self.refiner_out.weight)
        nn.init.zeros_(self.refiner_out.bias)
        self._normalize_decoder()

    def encode_score_value(self, h):
        x = self.encoder(h)
        return self.score_head(x), self.value_head(x)

    def encode(self, h):
        """Return the unmasked signed values for diagnostics/strict warmup."""
        return self.encode_score_value(h)[1]

    @torch.no_grad()
    def _normalize_decoder(self):
        weights = self.direct.weight
        norms = weights.squeeze(-1).squeeze(-1).norm(dim=0, keepdim=True)
        weights.div_(norms.view(1, -1, 1, 1).clamp_min(1e-8))

    def decode_strict(self, z):
        return self.direct(z) + self.dec_bias

    def decode(self, z, support=None):
        direct = self.decode_strict(z)
        correction = self.refiner_out(self.refiner_blocks(F.gelu(self.refiner_in(z))))
        return direct + correction


class PracticalContextVectorStrictSAE(nn.Module, SparseMessageMixin):
    """Strict direct-vector path used as the practical-vector parent."""
    def __init__(self, C, K, d=None):
        super().__init__()
        self.C, self.K, self.d = C, K, d or min(2 * C, 256)
        self.encoder = ContextualFieldEncoder(C, self.d, (1, 2, 4, 1))
        self.score_head = nn.Conv2d(self.d, K, 1)
        self.value_head = nn.Conv2d(self.d, K, 1)
        self.direct = nn.Conv2d(K, C, 1, bias=False)
        self.dec_bias = nn.Parameter(torch.zeros(1, C, 1, 1))
        self._normalize_decoder()

    def encode_score_value(self, h):
        x = self.encoder(h)
        return self.score_head(x), self.value_head(x)

    def encode(self, h):
        return self.encode_score_value(h)[1]

    @torch.no_grad()
    def _normalize_decoder(self):
        weights = self.direct.weight
        norms = weights.squeeze(-1).squeeze(-1).norm(dim=0, keepdim=True)
        weights.div_(norms.view(1, -1, 1, 1).clamp_min(1e-8))

    def decode(self, z, support=None):
        return self.direct(z) + self.dec_bias


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


class ContextVectorSAE(nn.Module):
    """Vector SAE with a zero-init local encoder context branch.

    The decoder is deliberately the same 1x1 linear decoder as VectorSAE and
    receives only sparse coefficients. Spatial context is encoder-only.
    """
    def __init__(self, C, K):
        super().__init__()
        self.C, self.K = C, K
        self.pre_bias = nn.Parameter(torch.zeros(1, C, 1, 1))
        self.context = nn.Conv2d(C, C, 3, padding=1, groups=C, bias=False)
        nn.init.zeros_(self.context.weight)
        self.enc = nn.Conv2d(C, K, 1)
        self.dec = nn.Conv2d(K, C, 1, bias=False)
        self.dec_bias = nn.Parameter(torch.zeros(1, C, 1, 1))
        self._normalize_decoder()

    @torch.no_grad()
    def _normalize_decoder(self):
        self.dec.weight.copy_(self.dec.weight / (self.dec.weight.norm(dim=0, keepdim=True) + 1e-8))

    @torch.no_grad()
    def load_vector_state(self, vector):
        """Copy an ordinary VectorSAE state for an exact zero-context start."""
        self.pre_bias.copy_(vector.pre_bias)
        self.enc.load_state_dict(vector.enc.state_dict())
        self.dec.load_state_dict(vector.dec.state_dict())
        self.dec_bias.copy_(vector.dec_bias)
        self.context.weight.zero_()

    def encode(self, h):
        centered = h - self.pre_bias
        return F.relu(self.enc(centered + self.context(centered)))

    def decode(self, z):
        return self.dec(z) + self.dec_bias


def build_sae(sae_type, C, K, d=None, n_blocks=3, k=5):
    # Legacy aliases remain for historical experiments.  The redo entrypoints
    # use the explicit family IDs below so they cannot silently fall back to a
    # single ReLU coefficient head.
    if sae_type == "field_old":
        return FieldOldSAE(C, K, d=d, n_blocks=n_blocks, k=k)
    if sae_type == "field_strict":
        return StrictAdditiveFieldSAE(C, K, d=d)
    if sae_type == "field_recovery":
        return PracticalRecoveryFieldSAE(C, K, d=d)
    if sae_type == "vector_context":
        return PracticalContextVectorSAE(C, K, d=d)
    if sae_type == "vector_context_strict":
        return PracticalContextVectorStrictSAE(C, K, d=d)
    if sae_type == "field":
        return FieldSAE(C, K, d=d, n_blocks=n_blocks, k=k)
    if sae_type == "vector":
        return VectorSAE(C, K)
    if sae_type == "context_vector":
        return ContextVectorSAE(C, K)
    raise ValueError(sae_type)
