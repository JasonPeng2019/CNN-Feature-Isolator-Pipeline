"""Changed masking module for isolated experiments."""
import math
import torch


def mask_global_topk(z, m):
    B, K, H, W = z.shape
    flat = z.reshape(B, -1)
    m = min(m, flat.shape[1])
    thresh = flat.topk(m, dim=1).values[:, -1:].clamp_min(torch.finfo(flat.dtype).tiny)
    keep = flat >= thresh
    return (flat * keep).reshape(B, K, H, W), keep.reshape(B, K, H, W)


def mask_global_frac(z, frac):
    K, H, W = z.shape[1:]
    return mask_global_topk(z, max(1, int(round(frac * K * H * W))))


def _avgpool_factor(Hs, Ht):
    assert Hs % Ht == 0, f"source {Hs} not divisible by target {Ht}"
    return Hs // Ht


def mask_rf_topk(z, k_rf, target_hw, dilate=0):
    B, K, Hs, Ws = z.shape
    Ht, Wt = target_hw
    fh, fw = _avgpool_factor(Hs, Ht), _avgpool_factor(Ws, Wt)
    if dilate == 0:
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
            win = z[:, :, h0:h1, w0:w1]
            flat = win.reshape(B, -1)
            kk = min(k_rf, flat.shape[1])
            thr = flat.topk(kk, dim=1).values[:, -1:].clamp_min(torch.finfo(flat.dtype).tiny)
            keep[:, :, h0:h1, w0:w1] |= (win >= thr.view(B, 1, 1, 1))
    return z * keep, keep


def mask_rf_frac(z, rf_frac, target_hw, dilate=0):
    B, K, Hs, Ws = z.shape
    Ht, Wt = target_hw
    fh, fw = _avgpool_factor(Hs, Ht), _avgpool_factor(Ws, Wt)
    if dilate == 0:
        win = z.reshape(B, K, Ht, fh, Wt, fw).permute(0, 2, 4, 1, 3, 5).reshape(B, Ht, Wt, K * fh * fw)
        kk = max(1, int(math.ceil(rf_frac * win.shape[-1])))
        thr = win.topk(min(kk, win.shape[-1]), dim=-1).values[..., -1:].clamp_min(torch.finfo(win.dtype).tiny)
        keepw = win >= thr
        keep = keepw.reshape(B, Ht, Wt, K, fh, fw).permute(0, 3, 1, 4, 2, 5).reshape(B, K, Hs, Ws)
        return z * keep, keep
    keep = torch.zeros_like(z, dtype=torch.bool)
    for i in range(Ht):
        for j in range(Wt):
            h0, h1 = max(0, i * fh - dilate), min(Hs, (i + 1) * fh + dilate)
            w0, w1 = max(0, j * fw - dilate), min(Ws, (j + 1) * fw + dilate)
            win = z[:, :, h0:h1, w0:w1]
            flat = win.reshape(B, -1)
            kk = max(1, int(math.ceil(rf_frac * flat.shape[1])))
            thr = flat.topk(min(kk, flat.shape[1]), dim=1).values[:, -1:].clamp_min(torch.finfo(flat.dtype).tiny)
            keep[:, :, h0:h1, w0:w1] |= (win >= thr.view(B, 1, 1, 1))
    return z * keep, keep


def apply_mask(z, mask_cfg):
    t = mask_cfg["type"]
    if t == "global":
        if mask_cfg.get("m") is not None:
            return mask_global_topk(z, mask_cfg["m"])
        return mask_global_frac(z, mask_cfg["frac"])
    if t == "rf":
        return mask_rf_topk(z, mask_cfg["k_rf"], tuple(mask_cfg["target_hw"]), mask_cfg.get("dilate", 0))
    if t == "rf_frac":
        return mask_rf_frac(z, mask_cfg["rf_frac"], tuple(mask_cfg["target_hw"]), mask_cfg.get("dilate", 0))
    if t == "none":
        return z, torch.ones_like(z, dtype=torch.bool)
    raise ValueError(t)
