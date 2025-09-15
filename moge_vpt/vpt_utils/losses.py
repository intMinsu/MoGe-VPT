import torch
from typing import Dict, List, Optional, Iterable, Tuple, Literal

from moge_vpt.vpt_utils.sample import bilinear_sample_pointmap
from moge_vpt.vpt_utils.alignment import weighted_z_scale_shift


def _smooth_l1(err: torch.Tensor, beta: float) -> torch.Tensor:
    if beta <= 0:
        return err
    return torch.where(err < beta, 0.5 * (err ** 2) / beta, err - 0.5 * beta)

def anchor_affine_inv_loss(
        pts_pred_cam_bhw3: torch.Tensor,     # (B,H,W,3)
        xy_proj_bm2: torch.Tensor,           # (B,M,2) padded
        Xc_bm3: torch.Tensor,                # (B,M,3) padded
        w_bm: Optional[torch.Tensor],        # (B,M) padded
        valid_bm: torch.Tensor,              # (B,M) bool
        beta: float = 0.0, trunc: float = 1.0,
        depth_eps: float = 1e-3,
) -> Tuple[torch.Tensor, Dict[str, float], torch.Tensor, torch.Tensor]:
    """
    Fully batched anchor loss (no Python loop over views).
    Shared xyz scale (s) and z-shift (dz) per view.
    """

    B, H, W, _ = pts_pred_cam_bhw3.shape
    # sample predictions at anchor pixels
    pred_bm3 = bilinear_sample_pointmap(pts_pred_cam_bhw3, xy_proj_bm2)  # (B,M,3)
    # GT camera points
    gt_bm3 = Xc_bm3  # (B,M,3)

    # weights: (valid / depth) with clamp
    if w_bm is None:
        w = torch.ones_like(valid_bm, dtype=pred_bm3.dtype)
    else:
        w = w_bm.to(pred_bm3.dtype)
    zsafe = gt_bm3[..., 2].clamp_min(depth_eps)
    w = w * valid_bm * (1.0 / zsafe)
    # clamp extreme weights (per view)
    mean_w = (w.sum(-1, keepdim=True) / valid_bm.sum(-1, keepdim=True).clamp_min(1)).clamp_min(1e-6)
    w = torch.minimum(w, 10.0 * mean_w)

    # solve per-view s, dz (weighted)
    s_b, dz_b = weighted_z_scale_shift(pred_bm3[..., 2], gt_bm3[..., 2], w)

    # align prediction
    scale = s_b[:, None, None]  # (B,1,1)
    shift = dz_b[:, None]  # (B,1)

    pred_aligned = pred_bm3 * scale  # scales x,y,z
    pred_aligned[..., 2] += shift  # shifts z only -> (B,M)

    # loss
    err = (pred_aligned - gt_bm3).abs()                          # (B,M,3)
    l = _smooth_l1(err, beta=beta)
    loss_per_b = (w[..., None] * l).sum(dim=(1, 2)) / w.sum(dim=1).clamp_min(1e-6)     # (B,)
    loss = loss_per_b.mean()

    # metrics
    rel = (pred_aligned.detach() - gt_bm3).norm(dim=-1) / zsafe  # (B,M)
    rel = torch.where(valid_bm, rel, torch.zeros_like(rel))
    denom = valid_bm.sum().clamp_min(1)
    delta = (rel < 1).sum() / denom
    terr  = torch.clamp(rel, max=trunc).sum() / denom

    misc = dict(
        delta=float(delta.item()),
        trunc_err=float(terr.item()),
        avg_scale=float(s_b.mean().item()),
        avg_dz=float(dz_b.mean().item()),
    )
    return loss, misc, scale, shift
