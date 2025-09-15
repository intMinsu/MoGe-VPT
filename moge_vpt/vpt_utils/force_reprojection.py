import torch
from typing import Dict, List, Optional, Iterable, Tuple, Literal

def focal_diag_norm_from_fovx(
    aspect: torch.Tensor | float,
    fovx_deg: torch.Tensor | float,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """
    Diagonal-normalized focal (tensor version).

    Defines a single scalar parameter `focal_diag_norm` measured relative to the
    *half* image diagonal (in normalized image units), such that the per-axis
    normalized focals are recovered by:
        fx_norm = focal_diag_norm * (sqrt(1+a^2) / (2a))
        fy_norm = focal_diag_norm * (sqrt(1+a^2) / 2)
    where a = aspect = W/H.

    It is linked to the horizontal FOV via:
        focal_diag_norm = a / (sqrt(1 + a^2) * tan(FOVx/2))

    Parameters
    ----------
    aspect : float or Tensor
        Aspect ratio a = W/H. Can be scalar or broadcastable tensor.
    fovx_deg : float or Tensor
        Horizontal field-of-view in degrees. Scalar or broadcastable tensor.
    device : torch.device, optional
        Target device for the returned tensor (overrides inputs if given).
    dtype : torch.dtype, optional
        Target dtype for the returned tensor (overrides inputs if given).

    Returns
    -------
    torch.Tensor
        `focal_diag_norm` with shape = broadcast(aspect, fovx_deg), on the chosen device/dtype.

    Notes
    -----
    - All ops are differentiable.
    - Assumes pinhole camera, no skew, principal point at image center.
    - This parameterization is symmetric across aspect ratios and can be
      convenient for optimization; convert to (fx, fy) with the helpers below.
    """
    a = torch.as_tensor(aspect, device=device, dtype=dtype)
    theta = torch.deg2rad(torch.as_tensor(fovx_deg, device=a.device, dtype=a.dtype)) * 0.5
    # focal_diag_norm = a / (sqrt(1+a^2) * tan(theta))
    return a / (torch.sqrt(1 + a * a) * torch.tan(theta))


def fx_fy_from_focal_diag_norm(
    aspect: torch.Tensor | float,
    focal_diag_norm: torch.Tensor | float,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert diagonal-normalized focal to per-axis *normalized* focals (fx, fy).

    Given a = W/H and focal_diag_norm, returns:
        fx_norm = focal_diag_norm * (sqrt(1+a^2) / (2a))
        fy_norm = focal_diag_norm * (sqrt(1+a^2) / 2)

    Parameters
    ----------
    aspect : float or Tensor
        Aspect ratio a = W/H.
    focal_diag_norm : float or Tensor
        Output from `focal_diag_norm_from_fovx` (same broadcast rules).
    device, dtype : optional
        Target device/dtype for the outputs.

    Returns
    -------
    (fx_norm, fy_norm) : (Tensor, Tensor)
        Normalized focals, same broadcast shape.

    Notes
    -----
    - “Normalized” here means image coordinates in [0,1] (half-size = 0.5).
      To obtain *pixel* focals: fx_px = fx_norm * W, fy_px = fy_norm * H.
    """
    a = torch.as_tensor(aspect, device=device, dtype=dtype)
    f = torch.as_tensor(focal_diag_norm, device=a.device, dtype=a.dtype)
    s = torch.sqrt(1 + a * a)
    fx = f * (s / (2 * a))
    fy = f * (s / 2)
    return fx, fy


def fx_fy_from_fovx_norm(
    aspect: torch.Tensor | float,
    fovx_deg: torch.Tensor | float,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Directly get normalized (fx, fy) from FOVx and aspect ratio.

    Uses the standard pinhole relations:
        fx_norm = 0.5 / tan(FOVx/2)
        fy_norm = (a/2) / tan(FOVx/2)

    Parameters
    ----------
    aspect : float or Tensor
        Aspect ratio a = W/H.
    fovx_deg : float or Tensor
        Horizontal field-of-view in degrees.
    device, dtype : optional
        Target device/dtype.

    Returns
    -------
    (fx_norm, fy_norm) : (Tensor, Tensor)
        Normalized focals, broadcast to a common shape.
    """
    a = torch.as_tensor(aspect, device=device, dtype=dtype)
    theta = torch.deg2rad(torch.as_tensor(fovx_deg, device=a.device, dtype=a.dtype)) * 0.5
    inv_tan = 1.0 / torch.tan(theta)
    fx = 0.5 * inv_tan
    fy = (a * 0.5) * inv_tan
    return fx, fy