import os, re, ast, json
from pathlib import Path
from typing import Dict, List, Optional, Iterable, Tuple, Literal
import numpy as np
import torch
import utils3d
import click

from moge_vpt.vpt_utils.geometry import unproject_uvd_to_world, \
    intrinsics_to_fovx, affine_pts_to_cam
from moge_vpt.vpt_utils.losses import anchor_affine_inv_loss
from moge_vpt.vpt_utils.viz import visualize_xy_on_gray_fast,\
    save_pointclouds_bnc_to_ply_batched, save_pred_gt_clouds_bnc_to_ply_batched
from moge_vpt.vpt_utils.geometry import world_to_cam
from moge_vpt.vpt_utils.sample import cloud_to_depth_bhw_nearest, per_pixel_conf_front_mean
from moge_vpt.model.vpt_module import last_n_indices

# from moge_vpt.vpt_utils.force_reprojection import focal_diag_norm_from_fovx, fx_fy_from_focal_diag_norm
# from moge_vpt.vpt_utils.sample import plan_pixel_chunk_size, cloud_to_depth_bhw_hybrid_parallel_chunked
# from moge_vpt.utils.geometry_torch import normalized_view_plane_uv, recover_focal_shift


from gs_src.model.gaussian.cameras.cameras import Cameras

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

## Tiny helper for loading ##
def _load_per_view_ufm_files(scene_name: str, view_stem: str, k_nn: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    base = Path(scene_name) / f"ufm_{int(k_nn)}_nn" / view_stem
    cov = np.load(base / "per_view_covis.npy")                      # (H,W) float
    pcd  = np.load(base / "per_view_point_cloud.npz")
    xyz = pcd["xyz"]                                                 # (N,3)
    conf = pcd["conf"] if "conf" in pcd.files else np.ones((xyz.shape[0],), dtype=np.float32)
    return cov, xyz, conf

def _estimate_M_points_cam(
    pts_cam_bnc: torch.Tensor, K_b33: torch.Tensor, H: int, W: int, round_uv: bool = True
) -> int:
    """Rough count of in-bounds, in-front projected points (for planner)."""
    X, Y, Z = pts_cam_bnc.unbind(-1)
    valid = torch.isfinite(pts_cam_bnc).all(-1) & (Z > 0)
    Zs = Z.clamp(min=1e-12)
    u = K_b33[:, 0, 0].unsqueeze(1) * (X / Zs) + K_b33[:, 0, 2].unsqueeze(1)
    v = K_b33[:, 1, 1].unsqueeze(1) * (Y / Zs) + K_b33[:, 1, 2].unsqueeze(1)
    valid &= torch.isfinite(u) & torch.isfinite(v)
    ui = (u.round() if round_uv else u.floor()).long()
    vi = (v.round() if round_uv else v.floor()).long()
    inside = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    return int((valid & inside).sum().item())

def _pad_clouds_to_nan(
        Xw_list,
        conf_list,
        device,
        dtype=torch.float32):
    """
    Inputs:
        Xw_list:  list of (Ni,3)
        conf_list:list of (Ni,)
    Returns:
        Xws_pad:  (Vchunk, Nmax, 3) filled with NaN past Ni
        conf_pad: (Vchunk, Nmax)   filled with 0 past Ni
        lengths:  (Vchunk,)        original Ni per view (torch long)
    """
    Vc = len(Xw_list)
    Nmax = max(int(x.shape[0]) for x in Xw_list) if Vc else 0
    Xws_pad = torch.full((Vc, Nmax, 3), float('nan'), device=device, dtype=dtype)
    conf_pad = torch.zeros((Vc, Nmax), device=device, dtype=dtype)
    lengths  = torch.zeros(Vc, device=device, dtype=torch.long)

    for j, (x, conf) in enumerate(zip(Xw_list, conf_list)):
        n = x.shape[0]
        if n == 0: continue
        Xws_pad[j, :n] = x.to(device=device, dtype=dtype)
        conf_pad[j, :n] = conf.to(device=device, dtype=dtype)
        lengths[j] = n
    return Xws_pad, conf_pad, lengths


## Tiny helper for minibatch training ##
def _stack_views(minibatch: List[Dict], device: str):
    """Stack per-view fields with common H,W; keep anchors as list."""
    Hs = [it["H"] for it in minibatch]
    Ws = [it["W"] for it in minibatch]
    assert len(set(Hs)) == 1 and len(set(Ws)) == 1, "All views in a scene must share H,W."
    H, W = Hs[0], Ws[0]

    imgs = torch.stack([it["image"] for it in minibatch], dim=0).to(device)  # (B,3,H,W)
    Ks   = torch.stack([it["K"] for it in minibatch], dim=0).to(device)      # (B,3,3)
    Rs   = torch.stack([it["R"] for it in minibatch], dim=0).to(device)      # (B,3,3)
    ts   = torch.stack([it["t"] for it in minibatch], dim=0).to(device)      # (B,3)

    anchors = [it["anchors"] for it in minibatch]
    img_paths = [it["img_path"] for it in minibatch]
    return imgs, H, W, Ks, Rs, ts, anchors, img_paths

def _pad_anchors(anchors_list: List[Dict], device: str, dtype=torch.float32):
    """
    Pad variable-length anchors to a fixed M_max across the minibatch.

    Returns:
        xy_bm2:    (B, M, 2)
        Xw_bm3:    (B, M, 3)
        w_bm:      (B, M)
        valid_bm:  (B, M) bool
    """
    B = len(anchors_list)
    M_max = max(a["xy"].shape[0] for a in anchors_list) if B > 0 else 0
    if M_max == 0:
        # no anchors at all
        return (torch.empty(B, 0, 2, device=device, dtype=dtype),
                torch.empty(B, 0, 3, device=device, dtype=dtype),
                torch.empty(B, 0, device=device, dtype=dtype),
                torch.empty(B, 0, dtype=torch.bool, device=device))

    xy  = torch.full((B, M_max, 2), -1.0, device=device, dtype=dtype)
    Xw  = torch.zeros((B, M_max, 3), device=device, dtype=dtype)
    wts = torch.zeros((B, M_max), device=device, dtype=dtype)
    val = torch.zeros((B, M_max), device=device, dtype=torch.bool)

    for b, a in enumerate(anchors_list):
        m = a["xy"].shape[0]
        xy[b, :m] = a["xy"].to(device, dtype)
        Xw[b, :m] = a["Xw"].to(device, dtype)
        if "w" in a and a["w"] is not None:
            wts[b, :m] = a["w"].to(device, dtype)
        else:
            wts[b, :m] = 1.0
        val[b, :m] = True

    return xy, Xw, wts, val

def attach_ufm_and_depth_sort(
    batch: Dict,
    # World cloud feeding the pseudo-GT depth renderer:
    ufm_k_nn: int,
    device,
    max_anchors_per_view: int = 50_000,
    conf_pow: float = 1.0,
    view_chunk_size: int = 4,
) -> List[Dict]:
    """
    Render depth from per-view UFM world clouds (robust hybrid aggregator), then unproject
    valid pixels back to world. Applies per-pixel confidence from UFM 'conf' (front cluster),
    and a view-level covisibility scalar from 'cov'. Returns per-view dicts with anchors:
    {'Xw': (m,3), 'xy': (m,2), 'w': (m,), 'covis': (m,)}.
    """
    # ---- unpack batch ----
    V, _, H, W = batch["context"]["image"].squeeze(0).shape
    scene_name = batch["scene"][0]
    img_paths = [Path(p) for p in batch["context"]["img_path"][0]]

    cameras_b1 = batch["context"]["cameras"]
    cameras = Cameras(
        **{k: (v[0] if isinstance(v, torch.Tensor) and v.ndim > 0 else v) for k, v in cameras_b1.items()},
        idx=batch['context']['index'].squeeze(0),
        distortion_params=None,
    )

    per_view_items: List[Dict] = []

    # Pre-make pixel grid once
    u_grid = torch.arange(W, device=device)
    v_grid = torch.arange(H, device=device)
    V_idx, U_idx = torch.meshgrid(v_grid, u_grid, indexing="ij")  # (H,W): V_idx=v, U_idx=u

    # ---- process views in chunks to avoid replicating (N,3) too widely ----
    for s in range(0, V, view_chunk_size):
        e = min(V, s + view_chunk_size)
        Vc = e - s

        # Build per-view K, R, t (OpenCV world->cam) tensors for the chunk
        K_chunk = torch.stack([cameras[i].get_K()[:3, :3] for i in range(s, e)], dim=0).to(device, dtype=torch.float32)
        R_chunk = torch.stack([cameras[i].R for i in range(s, e)], dim=0).to(device, dtype=torch.float32)
        t_chunk = torch.stack([cameras[i].T for i in range(s, e)], dim=0).to(device, dtype=torch.float32)

        cov_list = []
        Xw_list = []
        conf_list = []

        for j, view_idx in enumerate(range(s, e)):
            view_stem = img_paths[view_idx].stem
            cov, Xw, conf = _load_per_view_ufm_files(scene_name, view_stem, ufm_k_nn)
            cov_list.append(torch.from_numpy(cov).to(device, dtype=torch.float32))
            Xw_list.append(torch.from_numpy(Xw).to(device, dtype=torch.float32))
            conf_list.append(torch.from_numpy(conf).to(device, dtype=torch.float32))

        # --- PAD ragged clouds to NaN/0 so we can batch the rasterizer ---
        Xws_pad, confs_pad, lengths = _pad_clouds_to_nan(
            Xw_list,
            conf_list,
            device,
            dtype=torch.float32
        )  # (Vc,Nmax,3), (Vc,Nmax)

        covs = torch.stack(cov_list, dim=0)
        Xws = Xws_pad
        confs = confs_pad

        # Transform world cloud to each camera in the chunk: (Vc,N,3)
        # x_cam = R x_world + t
        pts_cam_bnc = (Xws @ R_chunk.transpose(1, 2)) + t_chunk.unsqueeze(1)  # (Vc,N,3)

        # --- plan pixel chunk size from VRAM cap ---
        current_chunk = s//view_chunk_size+1
        max_chunk = V//view_chunk_size if V%view_chunk_size==0 else V//view_chunk_size+1
        print(f"[Depth sorting] Chunk {current_chunk}/{max_chunk} processing ...")

        # --- render depth for the chunk (simple nearest) ---
        depth_bhw, mask_bhw = cloud_to_depth_bhw_nearest(
            pts_cam_bnc=pts_cam_bnc,
            K_b33=K_chunk,
            H=H, W=W,
            round_uv=True,
            points_chunk_size=None,  # or an int if you want streaming
        )

        # Build anchors per view in the chunk
        for j, view_idx in enumerate(range(s, e)):
            dep = depth_bhw[j]  # (H,W)
            msk = mask_bhw[j]  # (H,W)
            if not msk.any():
                per_view_items.append(dict(
                    image=batch["context"]["image"].squeeze(0)[view_idx].to(device),
                    W=W, H=H,
                    img_path=str(img_paths[view_idx]),
                    K=K_chunk[j], R=R_chunk[j], t=t_chunk[j],
                    anchors=dict(Xw=torch.empty(0, 3, device=device),
                                 xy=torch.empty(0, 2, device=device),
                                 w=torch.empty(0, device=device)),
                    covis=covs[j], camidx=cameras[view_idx].idx,
                ))
                continue

            # Per-pixel confidence from UFM points (front cluster mean)
            conf_map = per_pixel_conf_front_mean(
                pts_cam_nc=pts_cam_bnc[j],
                conf_n=confs[j],
                K_33=K_chunk[j],
                H=H, W=W,
                round_uv=True,
                front_rel=0.03,
                front_abs=0.05
            )  # (H,W)

            # Collect valid pixels
            vv = V_idx[msk]  # (M,)
            uu = U_idx[msk]  # (M,)
            xy = torch.stack([uu, vv], dim=-1).to(torch.float32)  # (M,2)
            d = dep[msk].to(torch.float32)  # (M,)

            w = conf_map[msk].clamp_min(0)
            if conf_pow != 1.0:
                w = w.clamp_min(1e-8).pow(conf_pow)

            # Fallback if degenerate
            if float(w.sum().item()) <= 0.0:
                w = torch.ones_like(w)

            # Sample anchors
            M = d.numel()
            if M > max_anchors_per_view:
                probs = (w / w.sum()).clamp_min(1e-12)
                sel = torch.multinomial(probs, num_samples=max_anchors_per_view, replacement=False)
            else:
                sel = torch.arange(M, device=device)

            xy_sel = xy[sel]
            d_sel = d[sel]
            w_sel = w[sel]

            # Unproject to world
            Xw_sel = unproject_uvd_to_world(xy_sel, d_sel, K_chunk[j], R_chunk[j], t_chunk[j])

            per_view_items.append(dict(
                image=batch["context"]["image"].squeeze(0)[view_idx].to(device),  # (3,H,W)
                W=W, H=H,
                img_path=str(img_paths[view_idx]),
                K=K_chunk[j], R=R_chunk[j], t=t_chunk[j],
                anchors=dict(
                    Xw=Xw_sel,  # (m,3) world points
                    xy=xy_sel,  # (m,2) pixel coords
                    w=w_sel,  # (m,) weights
                ),
                covis=covs[j],  # keep covisibility vector for downstream use
                camidx=cameras[view_idx].idx,
            ))

        # For cloud_to_depth_bhw_hybrid_parallel_chunked output visualization
        # img_paths = [view["img_path"] for view in per_view_items]
        # view_stems = [Path(img_path).stem for img_path in img_paths]
        # scene_id = "dataset/in2n_refined/{PUT HERE}}"
        # out_dirs = [Path(scene_id) / "mogev2-vpt" / view_stem for view_stem in view_stems]
        #
        # for out_dir in out_dirs:
        #     out_dir.mkdir(parents=True, exist_ok=True)
        #
        # rasterized_Xws = [view["anchors"]["Xw"] for view in per_view_items]
        # rasterized_Xws = torch.stack(rasterized_Xws, dim=0)
        #
        # raterize_ply_paths = [out_dir / f"original+rasterize.ply" for out_dir in out_dirs]
        # save_pred_gt_clouds_bnc_to_ply_batched(
        #     pred_bnc=rasterized_Xws,
        #     gt_bnc=Xws,
        #     out_paths=raterize_ply_paths,
        #     mask_pred_bN=None,
        #     mask_gt_bN=None,
        #     parallel=True
        # )

    return per_view_items

# ---------------------------------------
# The per-scene VPT optimization main loop
# ---------------------------------------
def fit_vpt_per_scene(
        model,
        device,
        scene_id: str,
        per_view_items: List[Dict],
        steps: int = 1000,
        batch_size: int = 4,
        num_tokens: int = 2000,
        lr_prompt: float = 1e-3,
        lambda_anchor: float = 1.0,
        lambda_prompt_l2: float = 1e-4,
        use_known_intrinsics: bool = True,
        apply_mask: bool = True,
        grad_clip: float = 1.0,
        log_every: int = 25,
        precision: Literal["fp32", "fp16", "bf16"] = "bf16",
        grad_accum_steps: int = 1,
        use_grad_checkpoint: bool = True,
):
    """
    Test-time, per-scene refinement of MoGe-2 using lightweight adaptation (VPT + Deep VPT + KV-Prefix + LoRA).

    This routine optimizes only the configured adaptation parameters (scene prompts, deep prompts,
    KV prefixes, and/or LoRA adapters) while freezing the rest of the network. It runs a fully-batched,
    step-based loop that:
      1) Samples a minibatch of views from `per_view_items`;
      2) Runs MoGe-2 forward in mixed precision (optional) and with gradient checkpointing (optional);
      3) Converts affine pointmaps to camera space (optionally using known intrinsics);
      4) Computes the batched anchor loss `anchor_affine_inv_loss` (no Python loop over views);
      5) Adds a small L2 regularizer on the shallow scene prompt;
      6) Backpropagates and updates only the configured adaptation parameters.

    Parameters
    ----------
    model : MoGeModel
        A MoGe-2 model where `model.encoder` is a `VPTDINOv2Encoder` with optional Deep VPT, KV-Prefix, and LoRA configured.
    device : str or torch.device
        Compute device (e.g., "cuda").
    scene_id : str
        Unique identifier for the scene. A per-scene parameter set (prompts, KV prefixes) will be created/selected.
    per_view_items : List[Dict]
        List of per-view dictionaries. Each entry MUST contain:
          - "image": (3,H,W) float Tensor in [0,1] or normalized (will be used as-is)
          - "H","W": ints (shared across views in a scene)
          - "img_path": str (used for logging/outputs)
          - "K": (3,3) intrinsics Tensor (pixels)
          - "R": (3,3) rotation (world→cam, OpenCV)
          - "t": (3,)   translation (world→cam, OpenCV)
          - "anchors": dict with:
              * "xy": (m,2) pixel coords (x,y), 0-based
              * "Xw": (m,3) world points
              * "w":  (m,)  optional per-anchor weights (if missing, defaults to 1)
          - Optional: "covis", "camidx" (not used by the loss here)
    steps : int
        Number of optimization steps.
    batch_size : int
        Minibatch size per optimization step. Must be ≤ len(per_view_items).
    num_tokens : int
        Base ViT token budget used by MoGe-2 forward (e.g., 1200–3600).
    lr_prompt : float
        Learning rate for adaptation parameters (shallow/deep/KV); LoRA adapters use 0.1× this LR by default.
    lambda_anchor : float
        Weight on the anchor loss.
    lambda_prompt_l2 : float
        Weight on the shallow-prompt L2 regularizer `(prompt**2).mean()`.
    use_known_intrinsics : bool
        If True, uses K to compute a known FoV and only recovers the Z-shift; otherwise recovers both focal and shift.
    apply_mask : bool
        If True, passes the predicted mask to focal/shift recovery for robustness (mask is NOT used in the loss).
    grad_clip : float
        Global gradient clipping (ℓ2) for stability.
    log_every : int
        Print/visualize every `log_every` steps.
    precision : {"fp32","fp16","bf16"}
        Mixed-precision mode for forward/backward. fp16 uses GradScaler; bf16 is unscaled AMP.
    grad_accum_steps : int
        Gradient accumulation steps. (Reserve for large effective batch; current code computes one step per minibatch.)
    use_grad_checkpoint : bool
        If True, enables gradient checkpointing for encoder/neck/heads to reduce memory.

    Behavior / Outputs
    ------------------
    • Creates/activates per-scene parameters via `encoder.add_scene_prompts(scene_id)` and `set_active_scene(scene_id)`.
    • Builds optimizer groups from `encoder.trainable_prompt_params(scene_id)`:
        - shallow (global prompt)
        - deep (per-block prompts)
        - kv (KV prefixes for attention)
        - lora (A/B weights of all LoRA adapters)
      You can tune per-group LR or WD easily.
    • Computes and prints:
        - total loss, anchor loss, shallow-prompt L2
        - weighted z-scale/shift stats (avg s, avg dz), δ-metric, truncated error
        - optional gradient norms per group (you can add the helper shown in comments)
    • Writes lightweight visualizations (xy overlays, .ply clouds) into:
        {scene_id}/mogev2-vpt/{view_stem}/...

    Notes
    -----
    - The anchor loss is **fully batched** with padding; there is no Python loop over views.
    - For focal/shift recovery, known FoV mode uses K→FoVx and solves only the Z shift; otherwise solves both.
    - Only adaptation parameters (prompts, KV, LoRA) receive gradients; the backbone can remain frozen.
    - Make sure your `per_view_items` share the same (H,W) within a scene.

    Returns
    -------
    None
        Updates model parameters in-place and logs artifacts to disk.
    """

    use_amp = precision in ("fp16", "bf16")
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(precision, torch.float32)
    if precision == "fp16":
        scaler = torch.cuda.amp.GradScaler(enabled=(precision == "fp16"))
        raise NotImplementedError("fp16 training is not implemented yet - Implement GradScaler properly... or just use bf16 if you can.")

    if use_grad_checkpoint:
        model.encoder.enable_gradient_checkpointing()
        model.neck.enable_gradient_checkpointing()
        for head in ['points_head', 'normal_head', 'mask_head']:
            if hasattr(model, head):
                getattr(model, head).enable_gradient_checkpointing()

    if len(per_view_items) == 0:
        print(f"[{scene_id}] No views with UFM anchors—skipping.")
        return

    # Register & select prompts
    assert hasattr(model.encoder, "add_scene_prompts") and hasattr(model.encoder, "set_active_scene")
    model.encoder.add_scene_prompts(scene_id)
    model.encoder.set_active_scene(scene_id)

    # Freeze everything except prompts (optionally allow 1x1 heads too)
    # for p in model.parameters():
    #     p.requires_grad_(False)
    # prompt.requires_grad_(True)

    prompt = model.encoder.scene_prompts[scene_id]

    trainable_groups = model.encoder.trainable_prompt_params(scene_id)

    opt = torch.optim.AdamW(
        [
            {"params": trainable_groups.get("shallow", []), "lr": lr_prompt, "weight_decay": 0.0},
            {"params": trainable_groups.get("deep", []), "lr": lr_prompt, "weight_decay": 0.0},
            {"params": trainable_groups.get("kv", []), "lr": lr_prompt, "weight_decay": 0.0},
            {"params": trainable_groups.get("lora", []), "lr": lr_prompt*0.1, "weight_decay": 0.0},
        ], betas=(0.9, 0.999)
    )

    # minibatch sampler
    B = min(batch_size, len(per_view_items))

    # Simple ring-buffered indices for speed & determinism across steps
    order = np.arange(len(per_view_items))
    ptr = 0

    model.train()
    for step in range(1, steps + 1):
        # --- gather a minibatch without Python work per-example
        if ptr + B > len(order):
            np.random.shuffle(order)
            ptr = 0
        sel = order[ptr:ptr + B]; ptr += B
        views = [per_view_items[i] for i in sel]

        imgs, H, W, Ks, Rs, ts, anchors, img_paths = _stack_views(views, device)
        # pad anchors
        xy_bm2, Xw_bm3, w_bm, valid_bm = _pad_anchors(anchors, device=device, dtype=torch.float32)
        # print(f"w_bm : {w_bm}")
        # print(f"valid_bm : {valid_bm}")
        if xy_bm2.shape[1] == 0:
            # no anchors at all in this minibatch; skip step
            continue
        Xc_bm3 = world_to_cam(Rs, ts, Xw_bm3)

        # ---- forward (amp+batched) ----
        with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            out = model.forward(imgs, num_tokens=num_tokens)

        # fp32 conversion
        pointmap_pred_affine = out["points"].to(torch.float32) # (B,H,W,3)
        if apply_mask:
            mask = out.get("mask", None)
            mask.to(torch.float32)
        else:
            mask = None

        # camera-space conversion
        if use_known_intrinsics:
            fovx_batch_deg = intrinsics_to_fovx(Ks, W, unit="deg")   # (B,)
            pointmap_pred_cam = affine_pts_to_cam(
                pointmap_pred_affine, mask,
                W, H, known_fovx_deg=fovx_batch_deg).to(torch.float32)
        else:
            pointmap_pred_cam = affine_pts_to_cam(
                pointmap_pred_affine, mask,
                W, H, known_fovx_deg=None).to(torch.float32)

        loss_anchor, misc, scales, shifts = anchor_affine_inv_loss(
            pts_pred_cam_bhw3=pointmap_pred_cam,
            xy_proj_bm2=xy_bm2,
            Xc_bm3=Xc_bm3,
            w_bm=w_bm,
            valid_bm=valid_bm,
            beta=0.0,
            trunc=1.0
        )

        loss_reg = (prompt ** 2).mean()
        loss = lambda_anchor * loss_anchor + lambda_prompt_l2 * loss_reg

        # --- update
        opt.zero_grad(set_to_none=True)

        loss.backward()
        print(f"prompt: {prompt}")
        prompt_grad = float(prompt.grad.norm().item())
        deep_prompt_grad = torch.stack([p.grad for p in model.encoder._deep_prompts[scene_id].parameters()], dim=0)
        kv_prefix_grad = torch.stack([p.grad for p in model.encoder._kv_store[scene_id].parameters()], dim=0)
        print(f"Prompt grad : {prompt_grad}")

        print(f"Prompt shape : {prompt.shape}")
        print(f"Deep prompt shape : {deep_prompt_grad.shape}")
        print(f"KV prefix shape : {kv_prefix_grad.shape}")

        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_([prompt], max_norm=grad_clip)
        opt.step()

        if step % log_every == 0:
            gnorm = prompt_grad if (prompt.grad is not None) else 0.0
            print(
                f"[{scene_id}] step {step:05d} | "
                f"loss={float(loss):.4f}  anchor={float(loss_anchor):.4f}  reg={float(loss_reg):.4f} | "
                f"avg_s~{misc['avg_scale']:.3f} avg_dz~{misc['avg_dz']:.3f}  delta~{misc['delta']:.3f}  terr~{misc['trunc_err']:.3f} | "
                f"||grad(prompt)||={gnorm:.3e}"
            )

            # ---- Visualization ----
            view_stems = [Path(img_path).stem for img_path in img_paths]
            out_dirs = [Path(scene_id) / "mogev2-vpt" / view_stem for view_stem in view_stems]
            for out_dir in out_dirs:
                out_dir.mkdir(parents=True, exist_ok=True)

            if step // log_every == 1: # First time log, never log again
                xy_pngs = [out_dir / f"xy_proj_gt.png" for out_dir in out_dirs]
                visualize_xy_on_gray_fast(
                    image=imgs,
                    xy_pix=xy_bm2,
                    out_path=xy_pngs
                )

            affine_ply_paths = [out_dir / f"affine-pred-{step + 1}.ply" for out_dir in out_dirs]
            save_pointclouds_bnc_to_ply_batched(
                points_bnc=pointmap_pred_affine.reshape(B, -1 ,3),
                out_paths=affine_ply_paths,
                mask_bN=out['mask'].reshape(B, -1),
                colors_bnc=None,
                parallel=True
            )

            not_aligned_ply_paths = [out_dir / f"Xc+pred+not_aligned-{step + 1}.ply" for out_dir in out_dirs]
            save_pred_gt_clouds_bnc_to_ply_batched(
                pred_bnc=pointmap_pred_cam.reshape(B, -1 ,3),
                gt_bnc=Xc_bm3,
                out_paths=not_aligned_ply_paths,
                mask_pred_bN=None,
                mask_gt_bN=None,
                parallel=True
            )

            xc_scale_zshift_ply_paths = [out_dir / f"Xc+pred+scaled+zshifted-{step + 1}.ply" for out_dir in out_dirs]
            scale_zshift_points = pointmap_pred_cam.reshape(B, -1 ,3)
            scale_zshift_points *= scales
            scale_zshift_points[..., 2] += shifts

            save_pred_gt_clouds_bnc_to_ply_batched(
                pred_bnc=scale_zshift_points,
                gt_bnc=Xc_bm3,
                out_paths=xc_scale_zshift_ply_paths,
                mask_pred_bN=None,
                mask_gt_bN=None,
                parallel=True
            )



# -----------------
# Orchestrating main
# -----------------
def _parse_scenes_arg(s: str) -> List[str]:
    if not s:
        return []
    try:
        v = json.loads(s)
        if isinstance(v, list):
            return [str(x) for x in v]
        return [str(v)]
    except Exception:
        pass
    try:
        v = ast.literal_eval(s)
        if isinstance(v, (list, tuple)):
            return [str(x) for x in v]
    except Exception:
        pass
    return [t for t in re.split(r'[,\s]+', s.strip()) if t]

@click.command(help='MoGe-2 VPT per-scene training script')
@click.option('--dataset_root', type=click.Path(path_type=Path), default=Path("dataset/in2n_refined"),
              help='Dataset root directory (e.g., dataset/in2n_refined)')
@click.option('--dataset_type', type=click.Choice(['custom', 'colmap', 'nerf_synthetic', 'nerfstudio']),
              default='colmap', help='Dataset type/name')
@click.option('--dataset_scenes', type=str, default='',
              help='Scenes as JSON list or comma/space-separated string, e.g. '
                   '\'["face_test","bear_test"]\' or face_test,bear_test')

@click.option('--steps', type=int, default=100,
              help=' (e.g. 100')
@click.option('--log_every_steps', type=int, default=25,
              help=' (e.g. 25')
@click.option('--batch_size', type=int, default=2,
              help=' (e.g. 2')

# UFM anchor argument
@click.option('--ufm_k_nn', type=int, default=4,
              help='Retrieve {ufm_k_nn}-NN UFM point cloud. It will find {dataset_root}/{dataset_scenes}[batch_idx]/{ufm_{ufm_k_nn}_nn} (e.g. 4')
@click.option('--ufm_max_points_per_view', type=int, default=4096,
              help='(e.g. 50_000')


# MoGE-2 arguments
@click.option('--resolution_level', type=int, default=9,
              help=' (e.g. 9)')
@click.option('--num_tokens', type=int, default=None,
              help=' (e.g. 2000)')

@click.option('--lr_prompt', type=float, default=1e-4,
              help=' (e.g. 1e-4)')
@click.option('--lambda_anchor', type=float, default=1.0,
              help=' (e.g. 1.0)')
@click.option('--lambda_prompt_l2', type=float, default=1e-4,
              help=' (e.g. 1e-4)')

def main(
        dataset_root: Path,
        dataset_type: Literal["colmap"],     # "custom" | "colmap" | "nerf_synthetic" | "nerfstudio"
        dataset_scenes: str,   # '["scene_a","scene_b"]' or "scene_a,scene_b"
        ufm_k_nn: int,
        ufm_max_points_per_view: int,
        steps: int,
        log_every_steps: int,
        batch_size: int,
        resolution_level: int,
        num_tokens: int,
        lr_prompt: float,
        lambda_anchor: float,
        lambda_prompt_l2: float,
):
    # --- Resolve scenes & datamodule ---
    device = "cuda" if torch.cuda.is_available() else "cpu"

    scenes = _parse_scenes_arg(dataset_scenes)
    if not scenes:
        raise ValueError("No scenes provided. Use --dataset_scenes e.g. '[\"face_test\",\"bear_test\"]'")

    from gs_src.dataset.data_module import DataModule
    from gs_src.dataset import DataLoaderCfg, DatasetCfg

    dataset_cfg = DatasetCfg(root=dataset_root, type=dataset_type, scenes=scenes)
    data_loader_cfg = DataLoaderCfg()
    dm = DataModule(dataset_cfg=dataset_cfg, data_loader_cfg=data_loader_cfg, global_rank=0)
    dm.setup()
    loader = dm.train_dataloader()

    # --- Load MoGe-2 with VPT encoder ---
    from moge_vpt.model.v2 import MoGeModel
    from moge_vpt.model.vpt_module import VPTDINOv2Encoder

    model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal")
    vanilla_enc = model.encoder

    vpt_enc = VPTDINOv2Encoder(
        backbone="dinov2_vitl14",
        intermediate_layers=[5, 11, 17, 23],
        dim_out=1024,
        vpt_num_tokens=8,
        vpt_init="zeros",
        freeze_backbone=True
    )
    vpt_enc.backbone.eval()

    vpt_enc.backbone.load_state_dict(vanilla_enc.backbone.state_dict(), strict=True)
    for dst, src in zip(vpt_enc.output_projections, vanilla_enc.output_projections):
        dst.load_state_dict(src.state_dict(), strict=True)

    model.encoder = vpt_enc
    model.to(device)
    model.eval()

    # VPT-Deep & KV-Prefix(P-Tuning v2) & Lora on specified ViT layers
    blk_last4 = last_n_indices(model.encoder.backbone, 4)
    print(f"blk_last4: {blk_last4}")

    # VPT
    model.encoder.configure_deep_vpt(
        blocks=blk_last4,
        prompt_len=8,
        init="zeros",
    )

    # KV-Prefix
    model.encoder.configure_kv_prefix(
        blocks=blk_last4,
        Pkv=8,
    )

    # LoRA
    model.encoder.inject_lora_backbone(
        last_n=2,
        targets=("attn.qkv", "attn.proj"),
        r=4,
        alpha=8
    )

    for batch_idx, batch in enumerate(loader):
        # one batch == one scene
        scene_name = batch["scene"][0]
        print(f"[MoGe2-VPT] {batch_idx} scene: {scene_name}")

        # Move tensors to device
        for k, v in list(batch.items()):
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device, non_blocking=True)

        # Attach UFM anchors per view and per-pixel rasterization for per-pixel median depth sort
        per_view_items = attach_ufm_and_depth_sort(
            batch=batch,
            ufm_k_nn=ufm_k_nn,
            device=device,
            max_anchors_per_view=ufm_max_points_per_view,
            conf_pow=1.0,
            view_chunk_size=256,
        )

        num_tokens_range = [1200, 3600]
        # Determine the number of base tokens to use
        if num_tokens is None:
            min_tokens, max_tokens = num_tokens_range
            num_tokens = int(min_tokens + (resolution_level / 9) * (max_tokens - min_tokens))
            print(f"[MoGe2-VPT] num_tokens is not assigned. As resolution_level is {resolution_level}, we set num_tokens as {num_tokens}.")

        # Fit prompts per scene
        fit_vpt_per_scene(
            model=model,
            device=device,
            scene_id=scene_name,
            per_view_items=per_view_items,
            steps=steps,
            batch_size=batch_size,
            num_tokens=num_tokens,
            lr_prompt=lr_prompt,
            lambda_anchor=lambda_anchor,
            lambda_prompt_l2=lambda_prompt_l2,
            use_known_intrinsics=True,
            apply_mask=True,
            grad_clip=1.0,
            log_every=log_every_steps,
            precision="bf16",
            grad_accum_steps=1,
        )

        # (Optional) save scene-specific prompts/checkpoint
        if hasattr(model.encoder, "save_scene_prompts"):
            out_dir = Path(dataset_root) / "vpt_prompts"
            out_dir.mkdir(parents=True, exist_ok=True)
            model.encoder.save_scene_prompts(scene_name, out_dir)

if __name__ == "__main__":
    main()
