import torch
import torch.nn.functional as F
from typing import Optional, Literal, Tuple

from moge_vpt.vpt_utils.utils import points_to_bhw3

def _pix_to_norm_grid(xy_pix: torch.Tensor, W: int, H: int, dtype, device) -> torch.Tensor:
    """
    Pixel (x,y) → normalized grid for grid_sample with align_corners=False.

    xy_pix shape must be either:
      • (B, N, 2)          → returns (B, N, 1, 2)
      • (B, H_out, W_out, 2) → returns (B, H_out, W_out, 2)
    """
    if xy_pix.dim() not in (3, 4) or xy_pix.shape[-1] != 2:
        raise ValueError("xy_pix must be (B, N, 2) or (B, H_out, W_out, 2).")

    xy = xy_pix.to(device=device, dtype=dtype)
    scale = torch.tensor([W, H], device=device, dtype=dtype)
    grid = (xy + 0.5) / scale * 2 - 1  # x→[-1,1], y→[-1,1] for align_corners=False

    if grid.dim() == 3:                 # (B, N, 2) → (B, N, 1, 2)
        grid = grid.unsqueeze(2)
    return grid

def bilinear_sample_pointmap(
    batched_points: torch.Tensor,
    xy_pix: torch.Tensor,
    *,
    padding_mode: str = "border",
) -> torch.Tensor:
    """
    Bilinearly sample a per-pixel 3D point map at given pixel coordinates.

    Inputs:
        points: 3D point map in one of
                (B,H,W,3), (B,3,H,W), (H,W,3), (3,H,W).
                Non-batched inputs are treated as B=1.
        xy_pix: Pixel coordinates with the same batch size B as `points`.
                Shape is either:
                  • (B, N, 2)          → N arbitrary points per batch
                  • (B, H_out, W_out, 2) → regular sampling grid
                Coordinates are 0-based: x→right, y→down (top-left origin).

    Returns:
        Tensor with sampled 3D points:
          • (B, N, 3)          if xy_pix is (B, N, 2)
          • (B, H_out, W_out, 3) if xy_pix is (B, H_out, W_out, 2)

    Notes:
        • Uses F.grid_sample with align_corners=False and the matching normalization.
        • `padding_mode` in {"zeros", "border", "reflection"}; default "border".
        • Gradients flow to both the map and the coordinates.

    Example:
        >>> H, W = 4, 6
        >>> pts = torch.randn(H, W, 3)          # treated as B=1
        >>> xy  = torch.tensor([[[0., 0.], [W-1., H-1.]]])  # (1, N=2, 2)
        >>> out = bilinear_sample_pointmap(pts, xy)         # (1, 2, 3)
        >>> out.shape
        torch.Size([1, 2, 3])
    """
    # Normalize points to (B, H, W, 3)
    pts_bhw3 = points_to_bhw3(batched_points)
    B, H, W, C = pts_bhw3.shape
    if C != 3:
        raise ValueError("Point map must have 3 channels in the last dimension.")

    # Check xy batch size
    if xy_pix.shape[0] != B:
        raise ValueError(f"Batch size mismatch: points has B={B}, xy_pix has B={xy_pix.shape[0]}.")

    # Prepare tensors for grid_sample
    pts_nchw = pts_bhw3.permute(0, 3, 1, 2).contiguous()                 # (B, 3, H, W)
    grid = _pix_to_norm_grid(xy_pix, W, H, dtype=pts_nchw.dtype, device=pts_nchw.device)

    # Bilinear sampling (align_corners=False matches normalization above)
    samp = F.grid_sample(
        pts_nchw, grid.to(dtype=pts_nchw.dtype),
        mode="bilinear", padding_mode=padding_mode, align_corners=False
    )

    if xy_pix.dim() == 3:
        # (B, 3, N, 1) → (B, N, 3)
        return samp[:, :, :, 0].permute(0, 2, 1).contiguous()
    else:
        # (B, 3, H_out, W_out) → (B, H_out, W_out, 3)
        return samp.permute(0, 2, 3, 1).contiguous()

@torch.no_grad()
def cloud_to_depth_bhw_nearest(
    pts_cam_bnc: torch.Tensor,   # (B,N,3) in CAMERA coords (X,Y,Z), Z>0 is in front
    K_b33: torch.Tensor,         # (B,3,3) intrinsics
    H: int, W: int,
    round_uv: bool = True,
    points_chunk_size: Optional[int] = None,   # e.g. 2_000_000 for very large clouds; None = one pass
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fast z-buffer rasterization to per-pixel nearest depth.

    Returns:
        depth_bhw: (B,H,W) float with NaN where empty
        mask_bhw:  (B,H,W) bool (True where a depth was written)
    """
    device, dtype = pts_cam_bnc.device, pts_cam_bnc.dtype
    B, N, _ = pts_cam_bnc.shape
    HW = H * W

    # Split coordinates
    X, Y, Z = pts_cam_bnc.unbind(-1)                     # (B,N)
    valid = torch.isfinite(pts_cam_bnc).all(-1) & (Z > 0)

    # Project
    Zs = Z.clamp(min=1e-12)
    fx, fy = K_b33[:, 0, 0].unsqueeze(1), K_b33[:, 1, 1].unsqueeze(1)
    cx, cy = K_b33[:, 0, 2].unsqueeze(1), K_b33[:, 1, 2].unsqueeze(1)
    u = fx * (X / Zs) + cx
    v = fy * (Y / Zs) + cy
    valid &= torch.isfinite(u) & torch.isfinite(v)

    # Integer pixel bins
    ui = (u.round() if round_uv else u.floor()).long()
    vi = (v.round() if round_uv else v.floor()).long()
    inside = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    keep = (valid & inside).view(-1)                      # (B*N,)

    if not keep.any():
        depth = torch.full((B, H, W), float("nan"), device=device, dtype=dtype)
        mask  = torch.zeros((B, H, W), device=device, dtype=torch.bool)
        return depth, mask

    # Flatten kept points
    z_all   = Z.view(-1)[keep]                            # (M,)
    ui_all  = ui.view(-1)[keep]
    vi_all  = vi.view(-1)[keep]
    b_ids   = torch.arange(B, device=device).view(B,1).expand(B,N).reshape(-1)[keep]
    lin_all = vi_all * W + ui_all                         # (M,)
    glin    = b_ids * HW + lin_all                        # global pixel id in [0, B*HW)

    # Z-buffer via per-pixel min reduction
    INF = torch.tensor(torch.finfo(dtype).max, device=device, dtype=dtype)
    depth_flat = torch.full((B * HW,), INF, device=device, dtype=dtype)

    # Process in one or multiple passes to limit peak memory
    if points_chunk_size is None:
        depth_flat.scatter_reduce_(0, glin, z_all, reduce="amin", include_self=True)
    else:
        M = glin.numel()
        for s in range(0, M, int(points_chunk_size)):
            e = min(M, s + int(points_chunk_size))
            depth_flat.scatter_reduce_(0, glin[s:e], z_all[s:e], reduce="amin", include_self=True)

    # Finalize: reshape, mask, NaN out empties
    depth = depth_flat.view(B, H, W)
    mask  = depth.lt(INF)
    depth = torch.where(mask, depth, torch.full_like(depth, float("nan")))
    return depth, mask


def plan_pixel_chunk_size(
    vram_cap_mb: float,
    M_points: int,
    Kmax_guess: int = 6,
    work_dtype: Literal["fp16","fp32"] = "fp16",
    safety: float = 1.3,              # extra safety factor for allocator overhead / fragmentation
    reserve_mb: float = 512.0,        # leave headroom for model, activations, other tensors
) -> tuple[int, int]:
    """
    Compute a safe pixel_chunk_size for the hybrid cloud→depth rasterizer
    under a given VRAM cap.

    Args:
        vram_cap_mb: Total VRAM budget (MB) you allow for this operation.
        M_points: Estimated #in-bounds projected points (M) in the (sub)batch.
        Kmax_guess: Estimated per-pixel max multiplicity (e.g., 6–12).
        work_dtype: "fp16" or "fp32" for padded/sort work buffers (outputs remain in input dtype).
        safety: Extra multiplicative headroom for allocator/temps (>1.0).
        reserve_mb: MB to keep free for model/other tensors.

    Returns:
        pixel_chunk_size: Suggested #hit-pixels to process per chunk (Pc).
        M_global_max: Max M allowed by the global grouping buffers; if M_points > M_global_max,
                      reduce batch/sub-batch size before chunking pixels.
    """
    s = 2 if work_dtype == "fp16" else 4
    budget_bytes = (vram_cap_mb * 1024**2) / safety - reserve_mb * 1024**2
    if budget_bytes <= 0:
        return 1, 0

    # Max points allowed by global term alone:
    bytes_per_point_global = (2*s + 32)
    M_global_max = int(max(0, budget_bytes // bytes_per_point_global))

    # If current M exceeds this, reduce batch first:
    if M_points > M_global_max:
        # Pc can still be computed, but global will dominate → advise lowering batch.
        Pc = 1
        return Pc, M_global_max

    # Leftover for chunk work (after paying global memory)
    leftover = budget_bytes - M_points * bytes_per_point_global
    if leftover <= 0:
        return 1, M_global_max

    # Per-row upper bound for chunk work
    bytes_per_row = Kmax_guess * (4*s + 11)

    Pc = int(max(1, leftover // bytes_per_row))
    return Pc, M_global_max


@torch.no_grad()
def cloud_to_depth_bhw_hybrid_parallel_chunked(
    pts_cam_bnc: torch.Tensor,         # (B,N,3) in CAMERA coords
    K_b33: torch.Tensor,               # (B,3,3)
    H: int, W: int,
    round_uv: bool = True,
    # --- occlusion thresholds ---
    gap_rel: float = 0.06,
    gap_abs: Optional[float] = None,
    iqr_rel: Optional[float] = 0.10,
    # --- occluded behavior ---
    occluded_strategy: Literal["nearest", "front_median"] = "nearest",
    front_min_pts: int = 1,
    even_strategy: Literal["lower", "upper", "mean"] = "lower",
    return_occlusion_map: bool = False,
    # --- memory control ---
    pixel_chunk_size: Optional[int] = None,  # max #hit-pixels per chunk; auto if None
    target_mem_mb: Optional[float] = 512.0,  # soft budget for padded/sort buffers; used if pixel_chunk_size is None
    use_fp16_workbuf: bool = False,          # compute & sort in float16 to cut memory (final depth still in input dtype)
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Rasterize (B,N,3) camera-space points into per-pixel depth (B,H,W) with a
    hybrid aggregator: use median per pixel unless occlusion is detected, then
    switch to nearest (or front-cluster median). Processes all batches in a
    single pass and limits memory via pixel chunking.

    Args:
        pts_cam_bnc: (B,N,3) points in camera coordinates (X,Y,Z), Z>0 in front of camera.
        K_b33: (B,3,3) pinhole intrinsics matrices.
        H, W: Output image height/width in pixels.
        round_uv: If True, round projected (u,v) to nearest integer; else floor.
        gap_rel: Occlusion if max adjacent depth gap > gap_rel * median for a pixel.
        gap_abs: Optional absolute gap threshold (meters) OR None to disable.
        iqr_rel: Occlusion if IQR > iqr_rel * median; None to disable IQR test.
        occluded_strategy: "nearest" or "front_median" (median of front cluster).
        front_min_pts: Min #samples to trust front cluster median when used.
        even_strategy: Median tie-breaker for even counts: "lower"|"upper"|"mean".
        return_occlusion_map: If True, also returns a (B,H,W) occlusion-flag map.
        pixel_chunk_size: #hit-pixels to process per chunk. If None, uses target_mem_mb heuristic.
        target_mem_mb: Soft memory budget for work buffers when pixel_chunk_size is None.
        use_fp16_workbuf: If True, padded/sort buffers are float16 (saves ~2× memory).

    Returns:
        depth_bhw: (B,H,W) float depth with NaN where empty.
        mask_bhw: (B,H,W) bool, True where a depth was written.
        occ_bhw: (B,H,W) bool occlusion flags if return_occlusion_map=True; else None.

    Notes:
        • Peak memory ≈ global(M) + chunk(Pc, Kmax); use the planner to pick Pc safely.
        • If M_points > M_global_max from the planner, reduce batch or split input first.
    """
    device = pts_cam_bnc.device
    dtype_in = pts_cam_bnc.dtype
    B, N, _ = pts_cam_bnc.shape
    HW = H * W

    # ---------- 1) Project all batches at once ----------
    X, Y, Z = pts_cam_bnc.unbind(-1)                                   # (B,N)
    valid = torch.isfinite(pts_cam_bnc).all(-1) & (Z > 0)

    Zs = Z.clamp(min=1e-12)
    x = X / Zs
    y = Y / Zs

    fx, fy = K_b33[:, 0, 0].unsqueeze(1), K_b33[:, 1, 1].unsqueeze(1)
    cx, cy = K_b33[:, 0, 2].unsqueeze(1), K_b33[:, 1, 2].unsqueeze(1)
    u = fx * x + cx
    v = fy * y + cy
    valid &= torch.isfinite(u) & torch.isfinite(v)

    ui = (u.round() if round_uv else u.floor()).long()
    vi = (v.round() if round_uv else v.floor()).long()
    inside = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    keep = (valid & inside).view(-1)                                    # (B*N,)

    if not keep.any():
        depth = torch.full((B, H, W), float("nan"), device=device, dtype=dtype_in)
        mask  = torch.zeros((B, H, W), device=device, dtype=torch.bool)
        occ   = torch.zeros((B, H, W), device=device, dtype=torch.bool) if return_occlusion_map else None
        return depth, mask, occ

    z_all  = Z.view(-1)[keep]                                           # (M,)
    ui_all = ui.view(-1)[keep]
    vi_all = vi.view(-1)[keep]
    b_ids  = torch.arange(B, device=device).view(B, 1).expand(B, N).reshape(-1)[keep]
    lin_all  = vi_all * W + ui_all                                      # (M,)
    glin_all = b_ids * HW + lin_all                                     # global pixel id in [0, B*HW)

    # ---------- 2) Group by global pixel id ----------
    order  = torch.argsort(glin_all)                                    # O(M log M)
    glin_s = glin_all[order]
    z_s    = z_all[order]

    uniq, counts = torch.unique(glin_s, return_counts=True)             # length P (hit pixels)
    P = uniq.numel()
    if P == 0:
        depth = torch.full((B, H, W), float("nan"), device=device, dtype=dtype_in)
        mask  = torch.zeros((B, H, W), device=device, dtype=torch.bool)
        occ   = torch.zeros((B, H, W), device=device, dtype=torch.bool) if return_occlusion_map else None
        return depth, mask, occ

    # Prefix sums: segment starts in z_s for each uniq
    offs = counts.cumsum(0)
    starts = torch.zeros_like(offs)
    starts[1:] = offs[:-1]

    # Decide chunking over P (hit pixels)
    if pixel_chunk_size is None:
        # Heuristic: choose chunk so that P_chunk * Kmax_chunk * 2 bytes/4 bytes ~ target_mem_mb
        # We don't know Kmax_chunk yet, but we can approximate with global Kmax as an upper bound.
        Kmax_global = int(counts.max().item())
        bytes_per = 2 if use_fp16_workbuf else 4  # float16 or float32 work buffers
        # Factor ~3 accounts for z_pad, z_sorted, and a few temporaries.
        if target_mem_mb is None:
            target_mem_mb = 512.0
        denom = max(1, Kmax_global * bytes_per * 3)
        # number of elements that fit in budget:
        elems = int((target_mem_mb * 1024 * 1024) // denom)
        pixel_chunk_size = max(1024, min(P, elems))
    # Safety clamp
    pixel_chunk_size = int(max(1, min(P, pixel_chunk_size)))

    # Output buffers
    depth = torch.full((B, H, W), float("nan"), device=device, dtype=dtype_in)
    mask  = torch.zeros((B, H, W), device=device, dtype=torch.bool)
    occ_map = torch.zeros((B, H, W), device=device, dtype=torch.bool) if return_occlusion_map else None
    depth_flat = depth.view(B, -1)
    mask_flat  = mask.view(B, -1)
    occ_flat   = occ_map.view(B, -1) if occ_map is not None else None

    # Choose working dtype for padded/sort buffers
    work_dtype = torch.float16 if use_fp16_workbuf else dtype_in

    rows_all = torch.arange(P, device=device)

    # ---------- 3) Process hit-pixels in chunks to limit memory ----------
    for s in range(0, P, pixel_chunk_size):
        e = min(P, s + pixel_chunk_size)
        idx = torch.arange(s, e, device=device)     # pixel rows in this chunk
        cnt = counts[idx]                           # (Pc,)
        st  = starts[idx]
        Pc = idx.numel()
        Kmax_chunk = int(cnt.max().item())
        if Kmax_chunk == 0:
            continue

        # Build padded (Pc, Kmax_chunk) with +inf
        z_pad = torch.full((Pc, Kmax_chunk), float("inf"), device=device, dtype=work_dtype)
        # Mask of real entries per row
        cols = torch.arange(Kmax_chunk, device=device)
        mask_rows = cols.unsqueeze(0) < cnt.unsqueeze(1)   # (Pc, Kmax_chunk)

        # Vectorized write: flatten the target slice and place z values
        # Gather the contiguous segments from z_s for this chunk and write respecting row lengths
        # z_s segment for row i: z_s[ st[i] : st[i]+cnt[i] ]
        # Concatenate them and assign with boolean mask
        segs = [z_s[st[i]: st[i] + cnt[i]] for i in range(Pc)]  # list of 1D tensors (variable lengths)
        z_concat = torch.cat(segs).to(work_dtype)
        z_pad[mask_rows] = z_concat

        # Row-wise sort (ascending). nearest depth is col 0
        z_sorted, _ = torch.sort(z_pad, dim=1)

        # Median per row
        if even_strategy == "mean":
            k1 = (cnt - 1) // 2
            k2 = cnt // 2
            r = torch.arange(Pc, device=device)
            med = 0.5 * (z_sorted[r, k1] + z_sorted[r, k2]).to(dtype_in)
        elif even_strategy == "upper":
            k = (cnt // 2)
            med = z_sorted[torch.arange(Pc, device=device), k].to(dtype_in)
        else:
            k = (cnt - 1) // 2
            med = z_sorted[torch.arange(Pc, device=device), k].to(dtype_in)

        # Occlusion cues
        if Kmax_chunk > 1:
            diffs = z_sorted[:, 1:] - z_sorted[:, :-1]                 # (Pc, Kmax_chunk-1)
            valid_diff = cols[:Kmax_chunk-1].unsqueeze(0) < (cnt - 1).unsqueeze(1)
            diffs = torch.where(valid_diff, diffs, torch.full_like(diffs, float("-inf")))
            gmax, gidx = diffs.max(dim=1)
            gmax = gmax.clamp_min(0).to(dtype_in)
        else:
            gmax = torch.zeros(Pc, device=device, dtype=dtype_in)
            gidx = torch.zeros(Pc, device=device, dtype=torch.long)

        if iqr_rel is not None:
            q1_idx = ((cnt - 1).float() * 0.25).floor().long()
            q3_idx = ((cnt - 1).float() * 0.75).ceil().long()
            r = torch.arange(Pc, device=device)
            q1 = z_sorted[r, q1_idx].to(dtype_in)
            q3 = z_sorted[r, q3_idx].to(dtype_in)
            iqr = (q3 - q1).clamp_min(0)
        else:
            iqr = torch.zeros(Pc, device=device, dtype=dtype_in)

        thr_gap = med * gap_rel
        if gap_abs is not None:
            thr_gap = torch.maximum(thr_gap, torch.as_tensor(gap_abs, device=device, dtype=dtype_in))

        occ_gap = gmax > thr_gap
        occ_iqr = (iqr_rel is not None) & (iqr > med * iqr_rel)
        occluded = (cnt >= 2) & (occ_gap | occ_iqr)

        # Choose per-pixel depth
        nearest = z_sorted[:, 0].to(dtype_in)

        if occluded_strategy == "nearest":
            depth_row = torch.where(occluded, nearest, med)
        else:
            # Front-cluster split at largest gap; size = gidx+1
            front_size = (gidx + 1).clamp(min=1)
            kf = (front_size - 1) // 2
            r = torch.arange(Pc, device=device)
            front_med = z_sorted[r, kf].to(dtype_in)
            use_front = occluded & (front_size >= max(1, front_min_pts))
            depth_row = torch.where(use_front, front_med, med)

        # Scatter this chunk back to (B,HW)
        uniq_chunk = uniq[idx]
        b_chunk = (uniq_chunk // HW).long()
        lin_chunk = (uniq_chunk % HW).long()

        depth_flat.index_put_((b_chunk, lin_chunk), depth_row, accumulate=False)
        mask_flat.index_put_((b_chunk, lin_chunk), torch.ones_like(depth_row, dtype=torch.bool), accumulate=False)
        if return_occlusion_map:
            occ_flat.index_put_((b_chunk, lin_chunk), occluded, accumulate=False)

    return depth, mask, (occ_map if return_occlusion_map else None)

@torch.no_grad()
def per_pixel_conf_front_mean(
    pts_cam_nc: torch.Tensor,         # (N,3) camera points for a single view
    conf_n: torch.Tensor,             # (N,) confidence per 3D point
    K_33: torch.Tensor,               # (3,3)
    H: int, W: int,
    round_uv: bool = True,
    front_rel: float = 0.03,          # keep <= (1+rel)*zmin
    front_abs: float = 0.05,          # or <= zmin + abs (meters)
) -> torch.Tensor:
    """
    Returns (H,W) per-pixel confidence = mean(conf of front-cluster contributions).
    Efficient scatter-reduce; no per-pixel loops.
    """
    X, Y, Z = pts_cam_nc.unbind(-1)
    valid = torch.isfinite(pts_cam_nc).all(-1) & (Z > 0)
    if not valid.any():
        return torch.zeros((H, W), device=pts_cam_nc.device, dtype=conf_n.dtype)

    Z = Z[valid]; X = X[valid]; Y = Y[valid]; conf = conf_n[valid]
    Zs = Z.clamp(min=1e-12)
    u = K_33[0, 0] * (X / Zs) + K_33[0, 2]
    v = K_33[1, 1] * (Y / Zs) + K_33[1, 2]
    finite = torch.isfinite(u) & torch.isfinite(v)
    if not finite.any():
        return torch.zeros((H, W), device=pts_cam_nc.device, dtype=conf_n.dtype)

    u = u[finite]; v = v[finite]; Z = Z[finite]; conf = conf[finite]
    ui = (u.round() if round_uv else u.floor()).long()
    vi = (v.round() if round_uv else v.floor()).long()
    inside = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    if not inside.any():
        return torch.zeros((H, W), device=pts_cam_nc.device, dtype=conf_n.dtype)

    ui = ui[inside]; vi = vi[inside]; Z = Z[inside]; conf = conf[inside]
    lin = vi * W + ui
    HW = H * W

    # zmin per pixel
    zmin = torch.full((HW,), torch.finfo(Z.dtype).max, device=Z.device, dtype=Z.dtype)
    zmin = zmin.scatter_reduce(0, lin, Z, reduce="amin", include_self=True)

    # front cluster mask
    zref = zmin[lin]
    keep = (Z <= zref * (1.0 + front_rel)) | (Z <= zref + front_abs)
    if not keep.any():
        return torch.zeros((H, W), device=Z.device, dtype=conf.dtype)

    lin = lin[keep]; conf = conf[keep]
    # mean per pixel = sum / count
    conf_sum = torch.zeros((HW,), device=conf.device, dtype=conf.dtype)
    conf_cnt = torch.zeros((HW,), device=conf.device, dtype=conf.dtype)
    conf_sum = conf_sum.scatter_add(0, lin, conf)
    conf_cnt = conf_cnt.scatter_add(0, lin, torch.ones_like(conf))
    conf_mean = torch.zeros_like(conf_sum)
    nz = conf_cnt > 0
    conf_mean[nz] = conf_sum[nz] / conf_cnt[nz]
    return conf_mean.view(H, W)