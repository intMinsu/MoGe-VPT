import os
from PIL import Image, ImageDraw
import numpy as np
import torch
import trimesh
from typing import Dict, List, Optional, Iterable, Tuple, Literal, Union, Sequence
from pathlib import Path
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor

PathLike = Union[str, "os.PathLike[str]"]

def _to_gray_bhwc_uint8(image_b3hw: torch.Tensor) -> torch.Tensor:
    """
        Convert batch of RGB images (B,3,H,W) in [0,1] float/half
        to grayscale uint8 format (B,H,W,3).

        Args:
            image_b3hw: Tensor of shape (B,3,H,W), values in [0,1].

        Returns:
            Tensor of shape (B,H,W,3), dtype uint8, grayscale replicated
            across 3 channels.
    """
    # image_b3hw: (B,3,H,W) in [0,1] float/half
    r, g, b = image_b3hw[:, 0], image_b3hw[:, 1], image_b3hw[:, 2]
    gray = 0.299 * r + 0.587 * g + 0.114 * b                    # (B,H,W)
    gray3 = gray.unsqueeze(-1).repeat(1, 1, 1, 3)               # (B,H,W,3)
    return (gray3.clamp(0, 1) * 255).round().to(torch.uint8)

def _normalize_batch(
    image: torch.Tensor,
    xy_pix: Union[torch.Tensor, Sequence[torch.Tensor]],
    out_path: Union[PathLike, Sequence[PathLike]],
):
    """
    Normalize heterogeneous batch inputs to consistent shapes and types.

    Args:
        image: RGB image(s) in one of:
            (3,H,W), (H,W,3), (B,3,H,W), or (B,H,W,3).
        xy_pix: Pixel coordinates per image, as either:
            (M,2) for single image, (B,M,2) for batch, or a sequence of Tensors
            where each is (Mi,2). Values are cast to float32 on `image`'s device.
        out_path: Output path(s). String/PathLike for single image, or a sequence
            of length B for batches.

    Returns:
        image_b3hw: Tensor of shape (B,3,H,W).
        xy_list: List of length B with Tensors of shape (Mi,2), dtype float32.
        out_paths: List[str] of length B.
        shape: Tuple (B,H,W).

    Raises:
        ValueError: If input shapes/types are incompatible (e.g., length mismatch).
        AssertionError: If channel count is not 3.
    """
    if image.dim() == 3:
        if image.shape[0] == 3: # (3, H, W)
            image_b3hw = image.unsqueeze(0)
        if image.shape[-1] ==3: # (H, W, 3)
            image_b3hw = image.permute(2, 0 ,1).unsqueeze(0)
    elif image.dim() == 4:
        if image.shape[1] == 3: # (B, 3, H, W)
            image_b3hw = image
        if image.shape[-1] == 3: # (B, H, W, 3)
            image_b3hw = image.permute(0, 3, 1, 2)
    else:
        raise ValueError(f"image must be (3,H,W)/(H,W,3) or (B,3,H,W)/(B,H,W,3); got {tuple(image.shape)}")
    B, C, H, W = image_b3hw.shape
    assert C == 3, "image must have 3 channels"

    # out_paths
    if isinstance(out_path, (str, os.PathLike)):
        if B != 1:
            raise ValueError("For B>1, out_path must be a sequence of length B.")
        out_paths = [str(out_path)]
    else:
        out_paths = [str(p) for p in out_path]
        if len(out_paths) != B:
            raise ValueError(f"len(out_path) ({len(out_paths)}) != batch size ({B})")

    # xy list (each (Mi,2) float)
    if isinstance(xy_pix, (list, tuple)):
        if len(xy_pix) != B:
            raise ValueError(f"xy_pix list length ({len(xy_pix)}) != batch size ({B})")
        xy_list = [x.detach().to(dtype=torch.float32, device=image_b3hw.device) for x in xy_pix]
    elif torch.is_tensor(xy_pix):
        if xy_pix.dim() == 2:
            if B != 1:
                raise ValueError("xy_pix is (M,2) but B>1; provide (B,M,2) or a list.")
            xy_list = [xy_pix.detach().to(dtype=torch.float32, device=image_b3hw.device)]
        elif xy_pix.dim() == 3 and xy_pix.shape[0] == B and xy_pix.shape[-1] == 2:
            xy_list = [xy_pix[b].detach().to(dtype=torch.float32, device=image_b3hw.device) for b in range(B)]
        else:
            raise ValueError("xy_pix must be (M,2), (B,M,2), or list of (Mi,2).")
    else:
        raise ValueError("xy_pix must be a Tensor or a sequence of Tensors.")

    return image_b3hw, xy_list, out_paths, (B, H, W)

def _make_disk_kernel(radius: int, device: torch.device) -> torch.Tensor:
    """
    Create a binary disk-shaped convolution kernel.

    Args:
        radius: Nonnegative disk radius in pixels.
        device: Torch device for the returned tensor.

    Returns:
        Tensor of shape (1,1,k,k), dtype float32 on `device`,
        where k = 2*max(radius,0)+1. Ones inside the disk, zeros outside.
        (Not normalized.)

    Example:
        >>> import torch
        >>> import torch.nn.functional as F
        >>> B, H, W = 1, 7, 7
        >>> impulses = torch.zeros((B, 1, H, W))
        >>> impulses[0, 0, 3, 3] = 1.0  # point at (u=3, v=3)
        >>> k = _make_disk_kernel(radius=2, device=impulses.device)
        >>> mask = (F.conv2d(impulses, k, padding=2) > 0)  # (B,1,H,W) bool
        >>> mask.shape
        torch.Size([1, 1, 7, 7])
    """

    r = int(max(radius, 0))
    k = 2 * r + 1
    yy, xx = torch.meshgrid(
        torch.arange(-r, r + 1, device=device),
        torch.arange(-r, r + 1, device=device),
        indexing="ij",
    )
    disk = ((xx**2 + yy**2) <= (r * r)).to(torch.float32)  # (k,k)
    return disk.view(1, 1, k, k)  # conv weight

# --- fast visualizer ---
@torch.no_grad()
def visualize_xy_on_gray_fast(
    image: torch.Tensor,
    xy_pix: Union[torch.Tensor, Sequence[torch.Tensor]],
    out_path: Union[PathLike, Sequence[PathLike]],
    radius: int = 2,
    thickness: int = 2,          # 0 -> filled disk; otherwise ring thickness (px)
    color: tuple[int, int, int] = (0, 255, 0),
) -> None:
    """
    Draw 2D points on grayscale copies of images using a fast batched
    convolutional disk/ring overlay, then save to disk.

    Core method (fast path):
        1) Build a sparse impulse map per image with 1s at integer-rounded (u,v).
        2) Convolve once with a disk kernel to get the outer mask.
        3) (Optional) Convolve with a smaller disk and subtract to form a ring.
        4) Apply the mask in one shot on a uint8 grayscale canvas.
        This is fully vectorized with `conv2d` on GPU/CPU, avoiding per-point loops.

    Args:
        image: RGB tensor (3,H,W)/(H,W,3)/(B,3,H,W)/(B,H,W,3) in [0,1] float/half.
        xy_pix: Pixel coords per image: (M,2), (B,M,2), or sequence of (Mi,2) in (u,v).
        out_path: Output path or sequence of length B.
        radius: Disk radius in pixels.
        thickness: 0 for filled disk; >0 = ring thickness (px).
        color: RGB tuple (0–255) used for the overlay.

    Example:
        >>> # Single image, 3 points
        >>> import torch
        >>> H, W = 256, 320
        >>> img = torch.rand(3, H, W)                      # [0,1]
        >>> pts = torch.tensor([[50, 40], [159, 200], [10, 310]], dtype=torch.float32)
        >>> visualize_xy_on_gray_fast(img, pts, "vis.png", radius=3, thickness=1, color=(255,0,0))
    """
    image_b3hw, xy_list, out_paths, (B, H, W) = _normalize_batch(image, xy_pix, out_path)
    device = image_b3hw.device

    # Build sparse impulse maps (B,1,H,W) with ones at rounded (u,v)
    impulses = torch.zeros((B, 1, H, W), device=device, dtype=torch.float32)
    for b in range(B):
        if xy_list[b].numel() == 0:
            continue
        uv = xy_list[b]
        u = uv[:, 0].round().long()
        v = uv[:, 1].round().long()
        inside = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        if inside.any():
            u, v = u[inside], v[inside]
            impulses[b, 0, v, u] = 1.0  # vectorized index_put_

    # Convolve once (or twice for ring) to get mask(s)
    k_outer = _make_disk_kernel(radius, device)
    mask_outer = (F.conv2d(impulses, k_outer, padding=radius) > 0)  # (B,1,H,W) bool

    if thickness <= 0:
        ring = mask_outer
    else:
        inner_r = max(radius - thickness, 0)
        if inner_r == 0:
            # "ring" is just outer perimeter area (approx): subtract a 1px erosion
            k_inner = _make_disk_kernel(max(radius - 1, 0), device)
        else:
            k_inner = _make_disk_kernel(inner_r, device)
        mask_inner = (F.conv2d(impulses, k_inner, padding=k_inner.shape[-1] // 2) > 0)
        ring = mask_outer & (~mask_inner)

    ring = ring.squeeze(1)  # (B,H,W)

    # Compose grayscale canvas and colorize ring pixels
    canvas = _to_gray_bhwc_uint8(image_b3hw)                  # (B,H,W,3) uint8
    color_t = torch.tensor(color, dtype=torch.uint8, device=canvas.device)
    canvas[ring] = color_t                                    # broadcast to (...,3)

    # Save
    for b in range(B):
        arr = canvas[b].detach().cpu().numpy()
        out_p = out_paths[b]
        out_dir = os.path.dirname(out_p)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        Image.fromarray(arr).save(out_p)

# ---------- generic single-cloud batched saver (B,N,3) ----------
def _export_one_ply(pts_np: np.ndarray, cols_np: Optional[np.ndarray], path: PathLike) -> int:
    cloud = trimesh.points.PointCloud(pts_np, colors=cols_np)
    out_dir = os.path.dirname(str(path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    cloud.export(path)
    return int(pts_np.shape[0])

def save_pointclouds_bnc_to_ply_batched(
    points_bnc: torch.Tensor,                    # (B,N,3) float
    out_paths: Sequence[PathLike],               # len == B
    mask_bN: Optional[torch.Tensor] = None,      # (B,N) bool
    colors_bnc: Optional[torch.Tensor] = None,   # (B,N,3) uint8 or float[0,1]
    parallel: bool = False,
    max_workers: int = 4,
) -> list[int]:
    assert points_bnc.dim() == 3 and points_bnc.shape[-1] == 3, f"points must be (B,N,3) - detected shape {points_bnc.shape}"
    B, N, _ = points_bnc.shape
    assert len(out_paths) == B, f"len(out_paths) must equal B. len(out_paths)={len(out_paths)}, B={B}."
    if mask_bN is not None:
        assert mask_bN.shape == (B, N), f"masks must be (B,N) - detected shape {mask_bN.shape}"
    if colors_bnc is not None:
        assert colors_bnc.shape == (B, N, 3), f"colors_bnc must be (B,N,3) - detected shape {colors_bnc.shape}"

    payloads = []
    counts = [0] * B

    for b in range(B):
        pts = points_bnc[b]                                 # (N,3)
        valid = torch.isfinite(pts).all(dim=-1)            # (N,)
        if mask_bN is not None:
            valid &= mask_bN[b].to(torch.bool)

        pts = pts[valid]
        if pts.numel() == 0:
            # write empty file? usually better to raise
            raise ValueError(f"[b={b}] No valid points to save.")

        pts_np = pts.detach().cpu().numpy()
        cols_np = None
        if colors_bnc is not None:
            cols = colors_bnc[b][valid]
            if cols.dtype != torch.uint8:
                cols = (cols.float().clamp(0, 1) * 255.0).to(torch.uint8)
            cols_np = cols.detach().cpu().numpy()

        payloads.append((pts_np, cols_np, out_paths[b]))

    if parallel and B > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = [ex.submit(_export_one_ply, *p) for p in payloads]
            for i, f in enumerate(futs):
                counts[i] = f.result()
    else:
        for i, p in enumerate(payloads):
            counts[i] = _export_one_ply(*p)

    return counts


# ---------- pred-vs-gt combined saver (B,Np,3) & (B,Ng,3) ----------
def _export_pred_gt_one_ply(
    pts_pred_np: np.ndarray,
    pts_gt_np: np.ndarray,
    color_pred_u8: Tuple[int, int, int],
    color_gt_u8: Tuple[int, int, int],
    path: PathLike,
) -> Tuple[int, int, int]:
    n_pred = int(pts_pred_np.shape[0]) if pts_pred_np.size else 0
    n_gt   = int(pts_gt_np.shape[0]) if pts_gt_np.size else 0
    if n_pred + n_gt == 0:
        raise ValueError("No valid points (pred + gt) to save.")

    pts_list, cols_list = [], []
    if n_pred:
        pts_list.append(pts_pred_np)
        cols_list.append(np.tile(np.array(color_pred_u8, dtype=np.uint8), (n_pred, 1)))
    if n_gt:
        pts_list.append(pts_gt_np)
        cols_list.append(np.tile(np.array(color_gt_u8, dtype=np.uint8), (n_gt, 1)))

    pts  = np.concatenate(pts_list, axis=0)
    cols = np.concatenate(cols_list, axis=0)
    cloud = trimesh.points.PointCloud(pts, colors=cols)

    out_dir = os.path.dirname(str(path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    cloud.export(path)
    return int(pts.shape[0]), n_pred, n_gt

def save_pred_gt_clouds_bnc_to_ply_batched(
    pred_bnc: torch.Tensor,                      # (B,Np,3)
    gt_bnc: torch.Tensor,                        # (B,Ng,3)
    out_paths: Sequence[PathLike],               # len == B
    mask_pred_bN: Optional[torch.Tensor] = None, # (B,Np)
    mask_gt_bN: Optional[torch.Tensor] = None,   # (B,Ng)
    color_pred: Tuple[int, int, int] = (0, 128, 255),
    color_gt:   Tuple[int, int, int] = (255, 64, 64),
    parallel: bool = False,
    max_workers: int = 4,
) -> list[Tuple[int, int, int]]:
    assert pred_bnc.dim() == 3 and pred_bnc.shape[-1] == 3, "pred must be (B,Np,3)"
    assert gt_bnc.dim()   == 3 and gt_bnc.shape[-1]   == 3, "gt must be (B,Ng,3)"
    Bp, Np, _ = pred_bnc.shape
    Bg, Ng, _ = gt_bnc.shape
    if Bp != Bg:
        raise ValueError(f"Batch size mismatch: pred B={Bp}, gt B={Bg}")
    B = Bp
    assert len(out_paths) == B, "len(out_paths) must equal B"
    if mask_pred_bN is not None:
        assert mask_pred_bN.shape == (B, Np)
    if mask_gt_bN is not None:
        assert mask_gt_bN.shape == (B, Ng)

    payloads = []
    results: list[Tuple[int, int, int]] = [(0, 0, 0)] * B

    for b in range(B):
        p = pred_bnc[b]                                # (Np,3)
        g = gt_bnc[b]                                  # (Ng,3)
        valid_p = torch.isfinite(p).all(dim=-1)
        valid_g = torch.isfinite(g).all(dim=-1)
        if mask_pred_bN is not None:
            valid_p &= mask_pred_bN[b].to(torch.bool)
        if mask_gt_bN is not None:
            valid_g &= mask_gt_bN[b].to(torch.bool)

        p = p[valid_p]
        g = g[valid_g]

        p_np = p.detach().cpu().numpy() if p.numel() else np.empty((0, 3), dtype=np.float32)
        g_np = g.detach().cpu().numpy() if g.numel() else np.empty((0, 3), dtype=np.float32)

        payloads.append((p_np, g_np, color_pred, color_gt, out_paths[b]))

    if parallel and B > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = [ex.submit(_export_pred_gt_one_ply, *p) for p in payloads]
            for i, f in enumerate(futs):
                results[i] = f.result()
    else:
        for i, p in enumerate(payloads):
            results[i] = _export_pred_gt_one_ply(*p)

    return results