"""Evaluation metrics: reconstruction fidelity, downstream preservation, sparsity."""
import torch
import torch.nn.functional as F


@torch.no_grad()
def recon_metrics(H, Hhat):
    """Relative L2 error, cosine sim, fraction of variance unexplained (FVU)."""
    num = (H - Hhat).pow(2).sum()
    den = H.pow(2).sum().clamp_min(1e-12)
    rel_l2 = (num / den).sqrt().item()
    cos = F.cosine_similarity(H.flatten(1), Hhat.flatten(1), dim=1).mean().item()
    var = (H - H.mean(0, keepdim=True)).pow(2).sum().clamp_min(1e-12)
    fvu = (num / var).item()
    return {"rel_l2": rel_l2, "cos": cos, "fvu": fvu}


@torch.no_grad()
def sparsity_metrics(keep):
    """keep: bool (B,K,H,W). L0 per image and fraction of field retained."""
    B = keep.shape[0]
    l0 = keep.reshape(B, -1).sum(1).float()
    frac = l0 / keep[0].numel()
    return {"l0_mean": l0.mean().item(), "frac_retained": frac.mean().item()}


@torch.no_grad()
def downstream_metrics(model, section, Hhat_denorm, labels, orig_logits):
    """Splice reconstructed (denormalized) hidden map back into the frozen model.
    Returns top-1 acc of spliced model and KL(orig || spliced)."""
    logits = model.forward_from(section, Hhat_denorm)
    top1 = (logits.argmax(1) == labels).float().mean().item()
    logp = F.log_softmax(logits, dim=1)
    p_orig = F.softmax(orig_logits, dim=1)
    kl = F.kl_div(logp, p_orig, reduction="batchmean").item()
    agree = (logits.argmax(1) == orig_logits.argmax(1)).float().mean().item()
    return {"spliced_top1": top1, "kl_orig_spliced": kl, "pred_agree": agree}
