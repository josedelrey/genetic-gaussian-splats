from __future__ import annotations
import torch

# --------------------------------------------------------------------
# Config / exports
# --------------------------------------------------------------------
_DEV = 'cpu'

__all__ = [
    "render_splats_rgb",           # order-dependent OVER compositing (with per-splat alpha)
    "_render_single_into_over",
    "_DEV",
]

# --------------------------------------------------------------------
# (Optional) compile hint (PyTorch 2.1+). Uncomment if you want.
# import torch._dynamo
# torch._dynamo.config.suppress_errors = True
# --------------------------------------------------------------------

# --------------------------------------------------------------------
# Internal cache for full coordinate vectors (avoids arange per splat)
# --------------------------------------------------------------------
_FULL_CACHE = {
    "H": None,
    "W": None,
    "dev": None,
    "x": None,   # torch.arange(W, device=dev, dtype=torch.float32)
    "y": None,   # torch.arange(H, device=dev, dtype=torch.float32)
}

def _ensure_full_coords(H: int, W: int, dev: torch.device):
    """Cache and reuse 1D coordinate vectors for the given (H,W,dev)."""
    if (_FULL_CACHE["H"] != H) or (_FULL_CACHE["W"] != W) or (_FULL_CACHE["dev"] != dev):
        _FULL_CACHE["H"] = H
        _FULL_CACHE["W"] = W
        _FULL_CACHE["dev"] = dev
        _FULL_CACHE["x"] = torch.arange(W, device=dev, dtype=torch.float32)
        _FULL_CACHE["y"] = torch.arange(H, device=dev, dtype=torch.float32)
    return _FULL_CACHE["x"], _FULL_CACHE["y"]

# --------------------------------------------------------------------
# (Kept for compatibility, but not used in the hot path anymore)
# --------------------------------------------------------------------
def _build_L_from_logs(a_log: torch.Tensor, b_log: torch.Tensor, c_raw: torch.Tensor, dev) -> torch.Tensor:
    """
    Construct lower-triangular Cholesky L in pixel units from
    a_log, b_log (diagonal logs) and c_raw (signed shear):
        L = [[exp(a_log), 0],
             [c_raw,      exp(b_log)]]
    """
    l11 = torch.exp(a_log)
    l22 = torch.exp(b_log)
    l11 = torch.clamp(l11, min=1e-12)
    l22 = torch.clamp(l22, min=1e-12)
    return torch.stack((
        torch.stack((l11, torch.tensor(0.0, device=dev))),
        torch.stack((c_raw, l22))
    ), dim=0)  # [2,2]

def _aabb_from_L(L: torch.Tensor, k_sigma: float) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Conservative half-extents (pixels) for a k-sigma ellipse via column-wise |.| sums:
      hx = k * (|v1_x| + |v2_x|)
      hy = k * (|v1_y| + |v2_y|)
    Ensures at least 1px to keep a non-empty crop.
    """
    v1 = L[:, 0].abs()
    v2 = L[:, 1].abs()
    hx = k_sigma * (v1[0] + v2[0])
    hy = k_sigma * (v1[1] + v2[1])
    return torch.clamp(hx, min=1.0), torch.clamp(hy, min=1.0)

def _quadform_via_tri(qx: torch.Tensor, qy: torch.Tensor,
                      l11: torch.Tensor, l21: torch.Tensor, l22: torch.Tensor) -> torch.Tensor:
    """
    With Σ = L L^T and L lower-triangular, compute q^T Σ^{-1} q via solves:
        Solve L y = q  ⇒  y1 = qx/l11,  y2 = (qy - l21*y1)/l22
        quad = y1^2 + y2^2
    """
    y1 = qx / l11
    y2 = (qy - l21 * y1) / l22
    return y1 * y1 + y2 * y2

# --------------------------------------------------------------------
# Hot path (optimized) — NOW WITH ALPHA
# Expected indiv format (renderer space): [x, y, a_log, b_log, c_raw, r, g, b, a]
# Colors and alpha must be normalized to [0,1] before arriving here.
# --------------------------------------------------------------------
@torch.no_grad()
def _render_single_into_over(
    C_img: torch.Tensor,   # [H,W,3] in-place image (0..1)
    indiv: torch.Tensor,   # [9] = [x,y,a_log,b_log,c_raw,r,g,b,a] on same device/dtype as C_img
    H: int,
    W: int,
    k_sigma: float,
    dev: torch.device,
) -> None:
    """
    Porter-Duff OVER per splat with per-splat alpha:
      f = exp(-0.5 * q^T Σ^{-1} q) in [0,1]
      α_eff = α * f
      C <- (1 - α_eff) * C + α_eff * color
    """
    # Expect indiv already to be float32 on dev (done in render_splats_rgb)
    x, y, a_log, b_log, c_raw, rc, gc, bc, a = indiv

    # Center in pixels (clamped)
    cx = (x.clamp(0.0, 1.0) * (W - 1))
    cy = (y.clamp(0.0, 1.0) * (H - 1))

    # Lower-triangular params directly (no tiny 2x2 tensor)
    l11 = torch.exp(a_log).clamp_min(1e-12)
    l21 = c_raw
    l22 = torch.exp(b_log).clamp_min(1e-12)

    # AABB half extents (conservative bound)
    # L = [[l11, 0],[l21,l22]] ⇒ columns v1=(l11,l21), v2=(0,l22)
    hx = k_sigma * l11.abs()
    hy = k_sigma * (l21.abs() + l22.abs())

    # Ensure at least 1 px
    hx = torch.maximum(hx, torch.tensor(1.0, device=dev))
    hy = torch.maximum(hy, torch.tensor(1.0, device=dev))

    # Integer crop (inclusive)
    x0 = int(torch.clamp(cx - hx, 0, W - 1).item())
    x1 = int(torch.clamp(cx + hx, 0, W - 1).item())
    y0 = int(torch.clamp(cy - hy, 0, H - 1).item())
    y1 = int(torch.clamp(cy + hy, 0, H - 1).item())
    if x1 < x0 or y1 < y0:
        return

    # 1D coords (no meshgrid). Use cached full vectors.
    full_x, full_y = _ensure_full_coords(H, W, dev)
    xs = full_x[x0:x1+1]         # [wx]
    ys = full_y[y0:y1+1]         # [wy]

    # Solve L y = q where q = [qx, qy]^T and L = [[l11,0],[l21,l22]]
    y1_x  = (xs - cx) / l11                      # [wx]
    y2_yx = (ys - cy)[:, None]                   # [wy,1]
    y2_yx = (y2_yx - l21 * y1_x[None, :]) / l22  # [wy,wx]
    quad  = y1_x[None, :]**2 + y2_yx**2          # [wy,wx]
    f     = torch.exp(-0.5 * quad)[..., None]    # [wy,wx,1]

    # Color already normalized to 0..1 in render_splats_rgb
    color = torch.stack((rc, gc, bc))            # [3]

    # Effective alpha = per-splat alpha * gaussian falloff
    alpha_eff = (a.clamp(0.0, 1.0)) * f          # [wy,wx,1]

    # OVER compositing: C = (1 - α)C + α*color (in-place fused ops)
    region = C_img[y0:y1+1, x0:x1+1, :]
    region.mul_(1.0 - alpha_eff).add_(alpha_eff * color)

@torch.no_grad()
def render_splats_rgb(
    genome: torch.Tensor,  # [N,9] or [9] in renderer format (last channel is alpha 0..255 before normalization)
    H: int,
    W: int,
    *,
    k_sigma: float = 3.0,
    device: torch.device | None = None,
    background: tuple[float, float, float] = (1.0, 1.0, 1.0),  # white
) -> torch.Tensor:
    """
    Order-dependent OVER renderer with per-splat alpha:
        For each splat (in given order):
            f = Gaussian
            α_eff = α * f
            C <- (1 - α_eff) * C + α_eff * color
    Returns: rgb in [0,1], shape [H,W,3] on CPU.
    """
    dev = device or _DEV
    dev = torch.device(dev) if not isinstance(dev, torch.device) else dev

    # Prepare genome on device once (float32)
    if genome.ndim == 1:
        genome = genome.unsqueeze(0)
    genome = genome.to(dev, dtype=torch.float32)

    # Pre-normalize & clamp colors AND alpha (vectorized) to 0..1
    # genome[:, 5:8] are r,g,b; genome[:, 8] is alpha — all originally in 0..255
    genome[:, 5:9].clamp_(0.0, 255.0).div_(255.0)

    # Init image to background
    C_img = torch.empty((H, W, 3), device=dev, dtype=torch.float32)
    C_img[:] = torch.tensor(background, device=dev, dtype=torch.float32)

    # Prebuild full coordinate vectors for this (H,W,dev)
    _ensure_full_coords(H, W, dev)

    # Render in order
    for i in range(genome.shape[0]):
        _render_single_into_over(C_img, genome[i], H, W, k_sigma, dev)

    return C_img.clamp_(0.0, 1.0).cpu()
