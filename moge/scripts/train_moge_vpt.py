import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

# ---------------------------
# Robust kernels
# ---------------------------
def charbonnier(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return torch.sqrt(x * x + eps)

def huber(x: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    ax = x.abs()
    return torch.where(ax < delta, 0.5 * x * x, delta * (ax - 0.5 * delta))

# ---------------------------
# Weighted mean/cov whitening
# ---------------------------
def weighted_mean_cov(X: torch.Tensor, w: torch.Tensor, eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    X: (M,3), w: (M,)
    returns: (mu, C_inv_sqrt) where C_inv_sqrt @ C_inv_sqrt^T ≈ C^{-1}
    """
    w = w.clamp_min(1e-8)
    Wsum = w.sum()
    mu = (w[:, None] * X).sum(0) / Wsum
    Xc = X - mu
    # weighted covariance
    C = (w[:, None, None] * (Xc[:, :, None] @ Xc[:, None, :])).sum(0) / Wsum
    # inverse sqrt via SVD (stable)
    U, S, Vh = torch.linalg.svd(C + eps * torch.eye(3, device=X.device, dtype=X.dtype))
    C_inv_sqrt = U @ torch.diag((S + eps).rsqrt()) @ Vh
    return mu, C_inv_sqrt

def affine_invariant_loss(
    X_pred: torch.Tensor,  # (M,3)
    X_tgt:  torch.Tensor,  # (M,3)
    w:      torch.Tensor,  # (M,)
    robust: str = "charb",
    eps: float = 1e-6,
) -> torch.Tensor:
    """MoGe-style affine-invariant point map loss (shape only)."""
    mu_p, Wp = weighted_mean_cov(X_pred, w, eps)
    mu_t, Wt = weighted_mean_cov(X_tgt,  w, eps)
    Xp_n = (X_pred - mu_p) @ Wp.T
    Xt_n = (X_tgt  - mu_t) @ Wt.T
    d = torch.linalg.norm(Xp_n - Xt_n, dim=-1)
    if robust == "charb": d = charbonnier(d, eps)
    elif robust == "huber": d = huber(d, delta=1.0)
    return (w * d).sum() / w.clamp_min(1e-8).sum()

# ---------------------------
# Weighted Umeyama (similarity)
# ---------------------------
def weighted_umeyama(
    X: torch.Tensor, Y: torch.Tensor, w: torch.Tensor, eps: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Find s,R,t minimizing || s R X + t - Y ||_2 with weights w.
    Returns s (scalar), R (3x3), t (3,)
    """
    w = w.clamp_min(1e-8)
    Wsum = w.sum()
    muX = (w[:, None] * X).sum(0) / Wsum
    muY = (w[:, None] * Y).sum(0) / Wsum
    Xc = X - muX
    Yc = Y - muY
    Sigma = (w[:, None, None] * (Xc[:, :, None] @ Yc[:, None, :])).sum(0) / Wsum
    U, S, Vt = torch.linalg.svd(Sigma)
    R = Vt.T @ U.T
    if torch.linalg.det(R) < 0:
        Vt = Vt.clone()
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    varX = (w * (Xc * Xc).sum(-1)).sum() / Wsum
    s = (S.sum() / (varX + eps)).clamp_min(1e-8)
    t = muY - s * (R @ muX)
    return s, R, t

def similarity_aligned_residual(
    X_pred: torch.Tensor, X_tgt: torch.Tensor, w: torch.Tensor,
    use_point_to_plane: bool = False,
    tgt_normals: Optional[torch.Tensor] = None,  # (M,3), if p2pl
    robust: str = "charb",
) -> torch.Tensor:
    """
    Compute residual after solving similarity on the (detached) predictions:
      - point-to-point: || s R X_pred + t - X_tgt ||
      - point-to-plane: n_t^T (s R X_pred + t - X_tgt)
    The similarity (s,R,t) is detached (no-grad) to keep optimization well-posed.
    """
    with torch.no_grad():
        s, R, t = weighted_umeyama(X_pred.detach(), X_tgt, w)
    Xp = (s * (X_pred @ R.T)) + t  # (M,3)

    if use_point_to_plane and (tgt_normals is not None):
        r = (tgt_normals * (Xp - X_tgt)).sum(-1)
    else:
        r = torch.linalg.norm(Xp - X_tgt, dim=-1)

    if robust == "charb": r = charbonnier(r)
    elif robust == "huber": r = huber(r, delta=1.0)
    return (w * r).sum() / w.clamp_min(1e-8).sum()

# ---------------------------
# Bilinear sampling from BHWC points map
# ---------------------------
def sample_points3d(points_bhwc: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
    """
    points_bhwc: (B,H,W,3) predicted points (camera frame, affine space OK)
    xy: (B,M,2) pixel coords in *pixels*
    returns: (B,M,3)
    """
    B, H, W, _ = points_bhwc.shape
    # normalize for grid_sample (align_corners=False), add +0.5 pixel center
    grid = xy.clone()
    grid[..., 0] = (grid[..., 0] + 0.5) / W * 2 - 1
    grid[..., 1] = (grid[..., 1] + 0.5) / H * 2 - 1
    grid = grid.view(B, 1, -1, 2)  # (B,1,M,2)
    pts = points_bhwc.permute(0, 3, 1, 2)         # (B,3,H,W)
    samp = F.grid_sample(pts, grid, mode="bilinear", padding_mode="zeros", align_corners=False)  # (B,3,1,M)
    return samp[:, :, 0, :].permute(0, 2, 1).contiguous()  # (B,M,3)


@dataclass
class TTALossCfg:
    # curriculum / weights
    w_affine: float = 1.0         # shape-only (affine-invariant)
    w_sim: float = 1.0            # similarity-aligned metric residual (p2p or p2pl)
    w_reproj: float = 0.0         # optional 2D reprojection term (if neighbors provided)
    # options
    use_point_to_plane: bool = False
    robust: str = "charb"         # 'charb' or 'huber'
    # anchor sampling
    max_anchors_per_view: int = 4096
    # schedule
    warmup_iters: int = 50        # w_sim=0 during warmup
    total_iters: int = 200

def reprojection_loss_optional(
    X_pred_cam2: torch.Tensor,            # (B,M,3) predicted points in *neighbor* camera-2 frame
    xy2_obs: torch.Tensor,                # (B,M,2) observed pixels in neighbor view (if you have them)
    K2: torch.Tensor,                     # (B,3,3) intrinsics of neighbor
    w: torch.Tensor,                      # (B,M)
    robust: str = "charb",
    eps: float = 1e-6
) -> torch.Tensor:
    """
    Optional: only if you pass neighbor correspondences.
    Projects X_pred_cam2 and compares to observed (u2,v2).
    """
    B, M, _ = X_pred_cam2.shape
    X = X_pred_cam2
    z = X[..., 2].clamp_min(1e-6)
    x_n = X[..., 0] / z
    y_n = X[..., 1] / z
    fx = K2[:, 0, 0].view(B, 1)
    fy = K2[:, 1, 1].view(B, 1)
    cx = K2[:, 0, 2].view(B, 1)
    cy = K2[:, 1, 2].view(B, 1)
    u = fx * x_n + cx
    v = fy * y_n + cy
    err = torch.sqrt((u - xy2_obs[..., 0]) ** 2 + (v - xy2_obs[..., 1]) ** 2 + eps)
    if robust == "charb": err = charbonnier(err)
    elif robust == "huber": err = huber(err, delta=1.0)
    return (w * err).sum() / w.clamp_min(1e-8).sum()


def adapt_scene_with_vpt(
    model,                                # MoGeModel with encoder = VPTDINOv2Encoder(...)
    batch_images: torch.Tensor,           # (B,3,H,W) Torch image batch for this scene
    num_tokens: int,                      # token budget for encoder
    anchors: Dict[str, torch.Tensor],     # dict of per-view anchors (you load/prepare these)
    cams: Optional[Dict[str, torch.Tensor]] = None,
    cfg: TTALossCfg = TTALossCfg(),
    lr: float = 1e-3
) -> Dict[str, float]:
    """
    Per-scene test-time adaptation optimizing ONLY VPT prompts.

    Expected `anchors` keys (you control loading; just map to these tensors):
      - 'xy':   (B,M,2) pixel coords for which you have 3D anchors (per view)
      - 'Xcam': (B,M,3) target 3D in that view's camera frame
      - 'w':    (B,M)   per-anchor weights (e.g., derived from per_view_covis.npy). If absent, use ones.
      - optional 'normals': (B,M,3) target normals for point-to-plane residual
      - optional 'nb': dict for reprojection term, if you have neighbor corr:
            'R_21': (B,3,3), 't_21': (B,3), transforms from view-1 cam to neighbor cam-2
            'K2':   (B,3,3) neighbor intrinsics
            'xy2':  (B,M,2) observed neighbor pixels corresponding to 'xy' in view-1
            'w2':   (B,M)   optional weights for reprojection term (fallback to 'w')

    Expected `cams` only if you compute reprojection another way (not needed otherwise).

    Returns: dict of averaged losses.
    """
    device = next(model.parameters()).device
    B, _, H, W = batch_images.shape

    # Train only VPT prompts
    for p in model.parameters(): p.requires_grad_(False)
    for p in model.encoder.parameters(): p.requires_grad_(False)
    vpt_params = [p for p in model.encoder.parameters() if p.requires_grad]  # may be empty if you froze earlier
    # Ensure prompts (shallow or deep) require grad:
    if hasattr(model.encoder, "prompt_shallow") and model.encoder.prompt_shallow is not None:
        model.encoder.prompt_shallow.requires_grad_(True)
        vpt_params.append(model.encoder.prompt_shallow)
    if hasattr(model.encoder, "prompt_deep"):
        for pp in model.encoder.prompt_deep:
            pp.requires_grad_(True)
        vpt_params += list(model.encoder.prompt_deep)

    assert len(vpt_params) > 0, "No trainable VPT params found. Ensure VPTDINOv2Encoder is used and prompts require_grad=True."

    optim = torch.optim.Adam(vpt_params, lr=lr)

    # Prepare anchors
    xy   = anchors['xy'].to(device)              # (B,M,2)
    Xcam = anchors['Xcam'].to(device)            # (B,M,3)
    w    = anchors.get('w', torch.ones_like(xy[..., 0])).to(device)  # (B,M)
    normals = anchors.get('normals', None)
    if normals is not None: normals = normals.to(device)

    # Optionally subsample anchors per view
    if cfg.max_anchors_per_view is not None:
        M = xy.shape[1]
        if M > cfg.max_anchors_per_view:
            # sample by weight
            with torch.no_grad():
                prob = (w / (w.sum(dim=1, keepdim=True) + 1e-8)).cpu()
                idx = torch.stack([torch.multinomial(prob[b], cfg.max_anchors_per_view, replacement=False)
                                   for b in range(B)], dim=0).to(device)
            def gather_bm(t):
                return t.gather(1, idx.unsqueeze(-1).expand(-1, -1, t.shape[-1])) if t.dim()==3 else t.gather(1, idx)
            xy   = gather_bm(xy)
            Xcam = gather_bm(Xcam)
            w    = gather_bm(w)

            if normals is not None:
                normals = gather_bm(normals)

    meter = {"L_aff": 0.0, "L_sim": 0.0, "L_reproj": 0.0}

    model.train()
    for it in range(cfg.total_iters):
        optim.zero_grad(set_to_none=True)

        # Forward (no autocast to keep numerics stable for SVDs)
        out = model.forward(batch_images, num_tokens=num_tokens)  # uses VPT prompts internally
        P = out["points"]  # (B,H,W,3) in camera frame but up to output remap (affine-friendly)

        # Sample predicted 3D at anchor pixels
        X_pred = sample_points3d(P, xy)  # (B,M,3)

        # Validity: keep anchors with positive depth in both pred & target (guards odd affine z)
        valid = (Xcam[..., 2] > 0) & (X_pred[..., 2] > 0)
        if valid.ndim == 2:
            mask = valid
        else:
            mask = valid.all(dim=-1)

        # Apply mask
        def mask_BM(t):
            return torch.where(mask[..., None], t, torch.nan) if t.dim()==3 else torch.where(mask, t, torch.nan)

        X_pred_m = mask_BM(X_pred)
        X_tgt_m  = mask_BM(Xcam)
        w_m      = mask_BM(w)

        # Flatten B,M -> (BM,*) with nans removed per batch
        def flatten_valid(*tensors):
            outs = []
            for b in range(B):
                m = (~torch.isnan(tensors[0][b]).any(dim=-1))
                out_b = [T[b][m] for T in tensors]
                outs.append(out_b)
            return [torch.cat([o[i] for o in outs], dim=0) for i in range(len(tensors))]

        Xp_flat, Xt_flat, w_flat = flatten_valid(X_pred_m, X_tgt_m, w_m)

        # 1) Affine-invariant (shape) loss
        L_aff = affine_invariant_loss(Xp_flat, Xt_flat, w_flat, robust=cfg.robust)

        # 2) Similarity-aligned metric residual (p2p or p2pl)
        if cfg.use_point_to_plane and (normals is not None):
            N_flat = flatten_valid(mask_BM(normals))[0]
            L_sim = similarity_aligned_residual(Xp_flat, Xt_flat, w_flat, use_point_to_plane=True, tgt_normals=N_flat, robust=cfg.robust)
        else:
            L_sim = similarity_aligned_residual(Xp_flat, Xt_flat, w_flat, use_point_to_plane=False, tgt_normals=None, robust=cfg.robust)

        # Warm-up schedule: no sim during warmup (stabilize shape)
        w_sim_eff = 0.0 if it < cfg.warmup_iters else cfg.w_sim

        # 3) Optional: 2D reprojection with neighbor views (only if provided)
        L_reproj = torch.tensor(0.0, device=device)
        if cfg.w_reproj > 0.0 and ('nb' in anchors):
            nb = anchors['nb']
            R21 = nb['R_21'].to(device)     # (B,3,3)
            t21 = nb['t_21'].to(device)     # (B,3)
            K2  = nb['K2'].to(device)       # (B,3,3)
            xy2 = nb['xy2'].to(device)      # (B,M,2)
            w2  = nb.get('w2', w).to(device)

            # Transform predicted points to cam2: X2 = R21 * X1 + t21
            # (broadcast over M)
            X2_pred = (X_pred @ R21.transpose(1,2)) + t21[:, None, :]
            L_reproj = reprojection_loss_optional(X2_pred, xy2, K2, w2, robust=cfg.robust)

        loss = (cfg.w_affine * L_aff) + (w_sim_eff * L_sim) + (cfg.w_reproj * L_reproj)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(vpt_params, 1.0)
        optim.step()

        # meters (EMA-style light smoothing)
        meter["L_aff"] = 0.9 * meter["L_aff"] + 0.1 * float(L_aff.detach().cpu())
        meter["L_sim"] = 0.9 * meter["L_sim"] + 0.1 * float(L_sim.detach().cpu())
        meter["L_reproj"] = 0.9 * meter["L_reproj"] + 0.1 * float(L_reproj.detach().cpu())

    return meter
