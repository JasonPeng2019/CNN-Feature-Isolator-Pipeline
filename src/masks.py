"""Masking = sparsity for the transformed coefficient field.

The mask is applied AFTER the encoder produces the coefficient field Z (B,K,H,W).
Hard TopK selection: gradients flow only through retained coefficients (the
others are zeroed), which is the standard TopK-SAE behaviour and needs no STE.

Two mask families (the doc's Mask Type 1 / Type 2):
  * global TopK over all (k,h,w) per image            -> mask_global_topk
  * receptive-field-local TopK within next-layer RF   -> mask_rf_topk
"""
import torch
import torch.nn.functional as F


def mask_global_topk(z, m):
    """Keep the top-`m` coefficients over all (K,H,W) per sample. z: (B,K,H,W)."""
    B, K, H, W = z.shape
    flat = z.reshape(B, -1)
    m = min(m, flat.shape[1])
    thresh = flat.topk(m, dim=1).values[:, -1:].clamp_min(torch.finfo(flat.dtype).tiny)
    keep = flat >= thresh
    return (flat * keep).reshape(B, K, H, W), keep.reshape(B, K, H, W)


def mask_global_frac(z, frac):
    """Global TopK specified as a fraction of the total field size."""
    K, H, W = z.shape[1:]
    return mask_global_topk(z, max(1, int(round(frac * K * H * W))))


def _avgpool_factor(Hs, Ht):
    """Integer stride/kernel mapping source grid Hs -> target grid Ht (RF approx)."""
    assert Hs % Ht == 0, f"source {Hs} not divisible by target {Ht}"
    return Hs // Ht


def mask_rf_topk(z, k_rf, target_hw, dilate=0):
    """Receptive-field-local TopK.

    For each location u in the next section's grid (target_hw = (Ht,Wt)), the
    receptive-field window in the current field z (B,K,Hs,Ws) is the block of
    source positions pooling to u, optionally dilated by `dilate` cells on each
    side to approximate a larger effective RF. Within all coeffs (k, p in RF(u))
    keep the top `k_rf`. Windows partition the field (plus optional overlap from
    dilation); overlapping keeps are unioned.
    """
    B, K, Hs, Ws = z.shape
    Ht, Wt = target_hw
    fh, fw = _avgpool_factor(Hs, Ht), _avgpool_factor(Ws, Wt)
    if dilate == 0:
        # vectorized: partition field into non-overlapping windows and TopK each.
        # (B,K,Ht,fh,Wt,fw) -> (B,Ht,Wt, K*fh*fw)
        win = z.reshape(B, K, Ht, fh, Wt, fw).permute(0, 2, 4, 1, 3, 5).reshape(B, Ht, Wt, K * fh * fw)
        kk = min(k_rf, win.shape[-1])
        thr = win.topk(kk, dim=-1).values[..., -1:].clamp_min(torch.finfo(win.dtype).tiny)
        keepw = win >= thr
        keep = keepw.reshape(B, Ht, Wt, K, fh, fw).permute(0, 3, 1, 4, 2, 5).reshape(B, K, Hs, Ws)
        return z * keep, keep
    keep = torch.zeros_like(z, dtype=torch.bool)
    for i in range(Ht):
        for j in range(Wt):
            h0, h1 = max(0, i * fh - dilate), min(Hs, (i + 1) * fh + dilate)
            w0, w1 = max(0, j * fw - dilate), min(Ws, (j + 1) * fw + dilate)
            win = z[:, :, h0:h1, w0:w1]                       # (B,K,wh,ww)
            flat = win.reshape(B, -1)
            kk = min(k_rf, flat.shape[1])
            thr = flat.topk(kk, dim=1).values[:, -1:].clamp_min(torch.finfo(flat.dtype).tiny)
            keep[:, :, h0:h1, w0:w1] |= (win >= thr.view(B, 1, 1, 1))
    return z * keep, keep


def apply_mask(z, mask_cfg):
    """Dispatch by config dict: {"type": "global", "m": int|None, "frac": float|None}
    or {"type": "rf", "k_rf": int, "target_hw": (Ht,Wt), "dilate": int}."""
    t = mask_cfg["type"]
    if t == "global":
        if mask_cfg.get("m") is not None:
            return mask_global_topk(z, mask_cfg["m"])
        return mask_global_frac(z, mask_cfg["frac"])
    if t == "rf":
        return mask_rf_topk(z, mask_cfg["k_rf"], tuple(mask_cfg["target_hw"]), mask_cfg.get("dilate", 0))
    if t == "none":
        return z, torch.ones_like(z, dtype=torch.bool)
    raise ValueError(t)
