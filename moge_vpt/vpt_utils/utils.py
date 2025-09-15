import torch
from typing import Optional, Sequence, Union

def points_to_bhw3(points: torch.Tensor) -> torch.Tensor:
    """
    Convert a point map to channel-last (B, H, W, 3).

    Accepts one of:
      • (B, H, W, 3) → returned as-is
      • (B, 3, H, W) → permuted to (B, H, W, 3)
      • (H, W, 3)    → expanded to (1, H, W, 3)
      • (3, H, W)    → permuted/expanded to (1, H, W, 3)

    Raises:
        ValueError: for unsupported shapes.
    """
    if points.dim() == 4:
        if points.shape[-1] == 3:               # (B, H, W, 3)
            return points
        if points.shape[1] == 3:                # (B, 3, H, W)
            return points.permute(0, 2, 3, 1).contiguous()
    elif points.dim() == 3:
        if points.shape[-1] == 3:               # (H, W, 3)
            return points.unsqueeze(0)
        if points.shape[0] == 3:                # (3, H, W)
            return points.permute(1, 2, 0).unsqueeze(0).contiguous()

    raise ValueError(f"Unsupported point-map shape {tuple(points.shape)}. "
                     "Use one of: (B,H,W,3), (B,3,H, W), (H,W,3), (3,H,W).")


def mask_to_bhw(mask: Optional[torch.Tensor], H: int, W: int, B: int) -> Optional[torch.Tensor]:
    """
    Normalize mask to (B, H, W) boolean or None. Accepts (B,H,W) or (H,W).
    """
    if mask is None:
        return None
    if mask.dim() == 3:
        assert mask.shape == (B, H, W), f"mask must be (B,H,W); got {tuple(mask.shape)}"
        return mask.to(torch.bool)
    if mask.dim() == 2:
        assert mask.shape == (H, W), f"mask must be (H,W); got {tuple(mask.shape)}"
        return mask.unsqueeze(0).expand(B, -1, -1).to(torch.bool)
    raise ValueError(f"Unsupported mask shape {tuple(mask.shape)}; use (B,H,W) or (H,W).")

def colors_to_bhw3(colors: Optional[torch.Tensor], H: int, W: int, B: int) -> Optional[torch.Tensor]:
    """
    Normalize colors to (B, H, W, 3) uint8 or None. Accepts (B,H,W,3), (B,3,H,W), (H,W,3), (3,H,W).
    """
    if colors is None:
        return None
    cols = points_to_bhw3(colors)
    assert cols.shape == (B, H, W, 3), f"colors must be (B,H,W,3); got {tuple(cols.shape)}"
    if cols.dtype != torch.uint8:
        cols = (cols.float().clamp(0, 1) * 255.0).to(torch.uint8)
    return cols