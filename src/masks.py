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


def standardize_selection_scores(scores, min_std=0.1):
    """Standardize each sample's flattened selection scores in FP32."""
    if scores.ndim < 2:
        raise ValueError("scores must include a batch dimension and field dimensions")
    flat = scores.float().reshape(scores.shape[0], -1)
    mean = flat.mean(dim=1, keepdim=True)
    # Population std is finite for a one-entry field and is sufficient for a
    # ranking surrogate.  The floor keeps temperature units well-conditioned.
    std = flat.std(dim=1, keepdim=True, unbiased=False).clamp_min(float(min_std))
    return ((flat - mean) / std).reshape_as(flat)


def _exact_topm_mask(flat_scores, m):
    """Return a deterministic exact-M mask and flattened indices."""
    if flat_scores.ndim != 2:
        raise ValueError("flat_scores must have shape (batch, candidates)")
    if int(m) < 1:
        raise ValueError(f"m must be positive, got {m}")
    count = min(int(m), flat_scores.shape[1])
    # A tiny index-order perturbation resolves equal FP32 scores while staying
    # below one ULP of the standardized score scale.  This is only a tie rule;
    # all ordinary score gaps retain their original ordering.
    scale = flat_scores.detach().abs().amax(dim=1, keepdim=True).clamp_min(1.0)
    epsilon = torch.finfo(torch.float32).eps * scale / max(1, flat_scores.shape[1])
    index = torch.arange(flat_scores.shape[1], device=flat_scores.device, dtype=torch.float32).view(1, -1)
    ranked = flat_scores + index * epsilon
    indices = ranked.topk(count, dim=1, sorted=False).indices
    hard = torch.zeros_like(flat_scores, dtype=torch.bool)
    hard.scatter_(1, indices, True)
    return hard, indices


def exact_topm(scores, m):
    """Select exactly ``m`` entries per sample from FP32 score logits."""
    flat = scores.float().reshape(scores.shape[0], -1)
    hard, indices = _exact_topm_mask(flat, m)
    return hard.reshape_as(scores), indices


def select_topm(values, scores, m, temperature=0.1):
    """Apply exact Top-M with a hard-forward straight-through selector.

    ``scores`` is an unconstrained selection head and is standardized per
    sample in FP32.  ``values`` is a separate signed value head.  The returned
    coefficient field is hard masked in the forward pass, while the detached
    sigmoid surrogate supplies finite selector gradients during training.
    """
    if values.shape != scores.shape:
        raise ValueError(f"values and scores must have equal shape, got {values.shape} and {scores.shape}")
    if float(temperature) <= 0:
        raise ValueError("temperature must be positive")
    score_flat = standardize_selection_scores(scores)
    hard_flat, indices = _exact_topm_mask(score_flat, m)
    count = hard_flat.shape[1] if int(m) >= hard_flat.shape[1] else int(m)
    # The threshold is detached deliberately: the sigmoid is a surrogate for
    # support selection, not an extra score objective.
    threshold = score_flat.topk(count, dim=1, sorted=True).values[:, -1:].detach()
    soft_flat = torch.sigmoid((score_flat - threshold) / float(temperature))
    ste_flat = hard_flat.to(torch.float32) + soft_flat - soft_flat.detach()
    selected = values.float().reshape(values.shape[0], -1) * ste_flat
    return selected.reshape_as(values.float()), hard_flat.reshape_as(values), indices


def encode_and_select(net, activation, m, temperature=0.1):
    """Encode a redo activation and apply its exact sparse-message operator.

    Redo families must expose ``encode_sparse`` so score logits and signed
    values remain independent.  Legacy models intentionally fail here rather
    than silently entering a scientifically different fallback path.
    """
    if not hasattr(net, "encode_sparse"):
        raise TypeError("redo SAE must expose independent score/value encode_sparse")
    return net.encode_sparse(activation, m, temperature=temperature)


def mask_global_topk(z, m):
    """Keep the top-`m` coefficients over all (K,H,W) per sample. z: (B,K,H,W)."""
    B, K, H, W = z.shape
    flat = z.reshape(B, -1)
    if m < 1:
        raise ValueError(f"m must be positive, got {m}")
    m = min(int(m), flat.shape[1])
    keep, _ = _exact_topm_mask(flat, m)
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
