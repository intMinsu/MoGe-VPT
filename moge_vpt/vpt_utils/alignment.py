import torch
from typing import Dict, List, Optional, Iterable, Tuple, Literal

def align_points_scale_z_shift(pred_pts: torch.Tensor,
                               gt_pts: torch.Tensor,
                               weight: Optional[torch.Tensor] = None,
                               eps: float = 1e-6):
    if weight is None:
        weight = torch.ones(pred_pts.shape[0], device=pred_pts.device, dtype=pred_pts.dtype)
    w = weight.clamp_min(1e-8)
    z_pred, z_gt = pred_pts[:,2], gt_pts[:,2]
    A = torch.stack([torch.sqrt(w) * z_pred, torch.sqrt(w) * torch.ones_like(z_pred)], dim=-1)
    b = torch.sqrt(w) * z_gt
    x = torch.linalg.lstsq(A, b).solution
    s_z, dz = x[0], x[1]
    xy_norm = torch.linalg.norm(pred_pts[:,:2], dim=-1) + eps
    if (xy_norm > 0).any():
        s_xy = ( (w * (gt_pts[:,:2] * pred_pts[:,:2]).sum(-1)).sum()
               / (w * (pred_pts[:,:2]**2).sum(-1)).sum().clamp_min(eps) )
        s = 0.7 * s_z + 0.3 * s_xy
    else:
        s = s_z
    shift = torch.tensor([0.0, 0.0, float(dz)], device=pred_pts.device, dtype=pred_pts.dtype)
    return s, shift

def weighted_z_scale_shift(zp_bm: torch.Tensor,
                            zg_bm: torch.Tensor,
                            w_bm: torch.Tensor,
                            eps: float = 1e-8) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Closed-form weighted regression of zg ≈ s * zp + dz (per batch element).

    zp_bm, zg_bm: (B,M)
    w_bm:         (B,M)  (already zeroed on invalid anchors)
    Returns:
        s_b:  (B,)  scale
        dz_b: (B,)  z-shift
    """
    B, M = zp_bm.shape
    sumw = w_bm.sum(dim=1, keepdim=True).clamp_min(eps)          # (B,1)
    mu_p = (w_bm * zp_bm).sum(dim=1, keepdim=True) / sumw        # (B,1)
    mu_g = (w_bm * zg_bm).sum(dim=1, keepdim=True) / sumw        # (B,1)

    zp_c = zp_bm - mu_p
    zg_c = zg_bm - mu_g

    var_p = (w_bm * zp_c * zp_c).sum(dim=1, keepdim=True).clamp_min(eps)  # (B,1)
    cov   = (w_bm * zp_c * zg_c).sum(dim=1, keepdim=True)                 # (B,1)

    s  = (cov / var_p).squeeze(1)                                         # (B,)
    dz = (mu_g - s[:, None] * mu_p).squeeze(1)                             # (B,)
    return s, dz