"""Phase 3 — sparse-code hierarchy (the headline).

Family I (independent SAEs + learned transition): freeze E_t,D_t,E_{t+1},D_{t+1};
learn T_t: Z~_t -> Z^_{t+1}, target the independently-learned Z_{t+1}=E_{t+1}(H_{t+1})
plus decoded reconstruction of H_{t+1}.

N4 frozen-block hybrid (causal upper bound, no learned transition): Z~_t -> D_t ->
H^_t -> frozen ResNet block(s) -> H^_{t+1} -> E_{t+1} -> Z^_{t+1}. Measures how well
the masked sparse code propagates through the *real* network.

Predictor handles spatial-resolution change between sections via strided convs.
"""
import torch, torch.nn as nn, torch.nn.functional as F


class TransitionPredictor(nn.Module):
    """Z_t (B,Kt,Hs,Ws) -> Z_{t+1} (B,Kt1,Ht,Wt). Strided conv matches resolution."""
    def __init__(self, Kt, Kt1, Hs, Ht, hidden=None, depth=1):
        super().__init__()
        hidden = hidden or max(Kt, Kt1)
        stride = Hs // Ht if Hs >= Ht else 1
        k = 3 if stride == 1 else (stride + 2)
        pad = k // 2
        layers = [nn.Conv2d(Kt, hidden, k, stride, pad), nn.GELU()]
        for _ in range(depth):
            layers += [nn.Conv2d(hidden, hidden, 3, 1, 1), nn.GELU()]
        layers += [nn.Conv2d(hidden, Kt1, 1)]
        self.net = nn.Sequential(*layers)
        self.up = (Hs < Ht)
        self.Ht = Ht

    def forward(self, z):
        out = self.net(z)
        if self.up or out.shape[-1] != self.Ht:
            out = F.interpolate(out, size=(self.Ht, self.Ht), mode="bilinear", align_corners=False)
        return F.relu(out)


def transition_loss(zhat, z_target, Hhat_next, H_next, beta=1.0):
    code = (zhat - z_target).pow(2).mean()
    recon = (Hhat_next - H_next).pow(2).sum() / (H_next.pow(2).sum() + 1e-8)
    return code + beta * recon, {"code_mse": code.item(), "recon_rel": recon.item()}


@torch.no_grad()
def frozen_block_hybrid(sae_t, sae_t1, model, src, dst, Hn_t, mean_t, std_t, mean_t1, std_t1, mask_fn):
    """N4: propagate masked sparse code at t through real frozen blocks to t+1.
    Returns predicted normalized Z_{t+1}, reconstructed raw H_{t+1}, for eval."""
    z = sae_t.encode(Hn_t); zt, _ = mask_fn(z); Hhat_t = sae_t.decode(zt)
    Hhat_t_raw = Hhat_t * std_t + mean_t
    Hhat_t1_raw = model.forward_between(src, dst, Hhat_t_raw)
    Hn_t1 = (Hhat_t1_raw - mean_t1) / std_t1
    z_t1 = sae_t1.encode(Hn_t1)
    return z_t1, Hhat_t1_raw
