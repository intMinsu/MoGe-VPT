import torch
from typing import Dict, List, Optional, Iterable, Tuple, Literal
import math

def world_to_cam(
        R_b33: torch.Tensor,
        t_b3: torch.Tensor,
        Xw_bm3: torch.Tensor
) -> torch.Tensor:
    """
    R_b33:   (B,3,3) world->cam
    t_b3:    (B,3)
    Xw_bm3:  (B,M,3)
    Return:  (B,M,3) camera-space
    """
    Xc = torch.einsum('bij,bmj->bmi', R_b33, Xw_bm3) + t_b3[:, None, :]
    return Xc

def unproject_uvd_to_world(
    uv_bm2: torch.Tensor,
    d_bm: torch.Tensor,
    K_b33: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor
) -> torch.Tensor:
    """
    Unproject pixel coordinates (u,v) with z-depth d into world coordinates.

    Model (OpenCV pinhole, column-vector convention):
        x_cam = d * K^{-1} [u, v, 1]^T
        x_w   = R^T (x_cam - t)
    This implementation uses row-vector code, so the last step is (x_cam - t) @ R.

    Args:
        uv_bm2: (M, 2) pixel coords in pixels (u to the right, v downward), top-left origin.
        d_bm:   (M,)   camera-space z-depths (NOT ray lengths).
        K_b33:  (3, 3) camera intrinsics.
        R:      (3, 3) world→camera rotation (so X_cam = R X_w + t).
        t:      (3,)   world→camera translation.

    Returns:
        x_w: (M, 3) world-space 3D points.

    Notes:
        • If your depth is ray length, use:
          dir_cam = (uv1 @ K^{-T}); dir_cam = dir_cam / dir_cam.norm(dim=-1, keepdim=True)
          x_cam   = d * dir_cam
        • Assumes right-handed camera with +Z forward as in OpenCV.

    """
    if not (uv_bm2.ndim == 2 and uv_bm2.shape[1] == 2):
        raise ValueError(f"uv_bm2 must be (M,2), got {tuple(uv_bm2.shape)}")
    if not (d_bm.ndim == 1 and d_bm.shape[0] == uv_bm2.shape[0]):
        raise ValueError("d_bm must be (M,) and match uv_bm2.shape[0].")
    if K_b33.shape != (3, 3) or R.shape != (3, 3) or t.shape != (3,):
        raise ValueError("K must be (3,3), R must be (3,3), t must be (3,)")

    # Keep everything on the same device/dtype
    device = uv_bm2.device
    dtype  = uv_bm2.dtype
    d_bm = d_bm.to(device=device, dtype=dtype)
    K_b33 = K_b33.to(device=device, dtype=dtype)
    R = R.to(device=device, dtype=dtype)
    t = t.to(device=device, dtype=dtype)

    KinvT = torch.linalg.inv(K_b33).T               # K^{-T}
    uv1 = torch.cat([uv_bm2, torch.ones(len(uv_bm2), 1, device=device, dtype=dtype)], dim=-1)  # (M,3)
    x_cam = (uv1 @ KinvT) * d_bm.unsqueeze(-1)      # (M,3) = d * ( [u v 1] K^{-T} )

    x_w = (x_cam - t) @ R                           # row-vector form of R^T (x_cam - t)
    return x_w

def intrinsics_to_fovx(
    K: torch.Tensor,
    W: int | torch.Tensor,
    unit: str = "deg",
) -> torch.Tensor:
    """
    Batched-safe conversion of intrinsics to horizontal FoV.

    Parameters
    ----------
    K : (..., 3, 3) torch.Tensor
        Camera intrinsics in pixels. Uses K[..., 0, 0] as fx.
    W : int or (...,) torch.Tensor
        Image width(s) in pixels. If tensor, it must be broadcastable to K's batch shape.
    unit : {"deg","rad"}
        Output unit.

    Returns
    -------
    torch.Tensor
        FoVx with shape equal to K's leading batch shape (i.e., K.shape[:-2]).
    """
    if not isinstance(K, torch.Tensor):
        raise TypeError("K must be a torch.Tensor")

    fx = K[..., 0, 0]  # (...,)

    W = torch.as_tensor(W, device=K.device, dtype=fx.dtype)
    if W.ndim == 0:
        W = W.expand_as(fx)        # broadcast scalar width to batch
    else:
        W = W.to(device=K.device, dtype=fx.dtype)
        W = torch.broadcast_to(W, fx.shape)

    fovx_rad = 2.0 * torch.atan((W * 0.5) / fx)
    if unit.lower() == "rad":
        return fovx_rad
    if unit.lower() == "deg":
        return torch.rad2deg(fovx_rad)
    raise ValueError("unit must be 'deg' or 'rad'.")

def affine_pts_to_cam(
    pointmap_bhw3: torch.Tensor,
    mask_bhw: torch.Tensor | None,
    W: int,
    H: int,
    known_fovx_deg: float | torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Convert affine MoGe point maps to camera space (batched).

    Parameters
    ----------
    pointmap_bhw3 : (B,H,W,3) or (H,W,3) float tensor
        Affine-space points from MoGe-2 forward().
    mask_bhw : (B,H,W) or (H,W) bool/float tensor or None
        Optional mask; used only to stabilize recover_focal_shift.
    W, H : int
        Image width/height (all views in the batch must share same size).
    known_fovx_deg : float or (B,) tensor or None
        If provided, bypass FoV estimation and only recover shift; else estimate both.

    Returns
    -------
    pts_cam : same shape as input
        Camera-space points (z shifted to camera coordinates).
    """
    from moge_vpt.utils.geometry_torch import recover_focal_shift

    # Normalize shapes to batched
    squeeze_batch = False
    if pointmap_bhw3.dim() == 3:
        pointmap_bhw3 = pointmap_bhw3.unsqueeze(0)          # -> (1,H,W,3)
        if mask_bhw is not None and mask_bhw.dim() == 2:
            mask_bhw = mask_bhw.unsqueeze(0)            # -> (1,H,W)
        squeeze_batch = True

    B, Hx, Wx, C = pointmap_bhw3.shape
    assert Hx == H and Wx == W, "All views in a batch must share the same H×W."

    mask_bin = (mask_bhw > 0.5) if (mask_bhw is not None) else None

    if known_fovx_deg is None:
        focal, shift = recover_focal_shift(pointmap_bhw3, mask_bin)  # focal: (B,), shift: (B,)
    else:
        aspect = W / H
        fovx = torch.as_tensor(known_fovx_deg, device=pointmap_bhw3.device, dtype=pointmap_bhw3.dtype)
        if fovx.ndim == 0:
            fovx = fovx.expand(B)
        focal = aspect / (1 + aspect**2) ** 0.5 / torch.tan(torch.deg2rad(fovx / 2))
        _, shift = recover_focal_shift(pointmap_bhw3, mask_bin, focal=focal)

    pts_cam = pointmap_bhw3.clone()
    pts_cam[..., 2] = pts_cam[..., 2] + shift[..., None, None]

    if squeeze_batch:
        pts_cam = pts_cam.squeeze(0)
    return pts_cam
