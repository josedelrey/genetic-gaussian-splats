from __future__ import annotations
import torch


_DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

__all__ = [
    "render_splats_rgb",           # NEW: weighted-average color blending
    "_render_single_into_avg",
    "_DEV",
]


def _build_L_from_logs(a_log: torch.Tensor, b_log: torch.Tensor, c_raw: torch.Tensor, dev) -> torch.Tensor:
    """
    Construct lower-triangular Cholesky L in pixel units from
    a_log, b_log (diagonal logs) and c_raw (signed shear):
        L = [[exp(a_log), 0],
             [c_raw,      exp(b_log)]]
    """
    l11 = torch.exp(a_log)
    l22 = torch.exp(b_log)
    eps = torch.tensor(1e-12, device=dev)
    l11 = torch.clamp(l11, min=eps)
    l22 = torch.clamp(l22, min=eps)
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
    qx, qy: [h,w] (pixels)
    l11, l21, l22: scalars (tensors)
    Returns: [h,w] quad values.
    """
    y1 = qx / l11
    y2 = (qy - l21 * y1) / l22
    return y1 * y1 + y2 * y2


@torch.no_grad()
def _render_single_into_avg(
    C_num: torch.Tensor,  # [H,W,3] accumulates sum(w * color)
    W_sum: torch.Tensor,  # [H,W,1] accumulates sum(w)
    indiv: torch.Tensor,  # [9] = [x, y, a_log, b_log, c_raw, r, g, b, alpha]
    H: int,
    W: int,
    k_sigma: float,
    dev: torch.device,
) -> None:
    """
    Add one splat using order-independent weighted-average blending.

    Genome row layout:
      [x, y, a_log, b_log, c_raw, r, g, b, alpha]
        - x, y in [0,1] (normalized coords, origin top-left)
        - a_log, b_log are logs of *pixel* scales (diagonal of L)
        - c_raw is signed shear in pixels (off-diagonal)
        - r, g, b in [0,255]
        - alpha in [0,1]
    """
    x, y, a_log, b_log, c_raw, rc, gc, bc, a = indiv.to(dev, dtype=torch.float32)

    # Clamp input ranges
    a  = torch.clamp(a,  0.0, 1.0)
    rc = torch.clamp(rc, 0.0, 255.0)
    gc = torch.clamp(gc, 0.0, 255.0)
    bc = torch.clamp(bc, 0.0, 255.0)

    # Center in pixels
    cx = torch.clamp(x, 0.0, 1.0) * (W - 1)
    cy = torch.clamp(y, 0.0, 1.0) * (H - 1)

    # Geometry
    L = _build_L_from_logs(a_log, b_log, c_raw, dev)
    hx, hy = _aabb_from_L(L, k_sigma)

    # Crop
    x0 = int(torch.clamp(cx - hx, 0, W - 1).item())
    x1 = int(torch.clamp(cx + hx, 0, W - 1).item())
    y0 = int(torch.clamp(cy - hy, 0, H - 1).item())
    y1 = int(torch.clamp(cy + hy, 0, H - 1).item())
    if x1 <= x0 or y1 <= y0:
        return  # off-screen / degenerate

    # Local grid
    xs = torch.arange(x0, x1 + 1, device=dev, dtype=torch.float32)
    ys = torch.arange(y0, y1 + 1, device=dev, dtype=torch.float32)
    X, Y = torch.meshgrid(xs, ys, indexing='xy')
    qx = X - cx
    qy = Y - cy

    # Gaussian weight (radially symmetric in Σ metric)
    l11 = L[0, 0]; l21 = L[1, 0]; l22 = L[1, 1]
    quad = _quadform_via_tri(qx, qy, l11, l21, l22)    # [h,w]
    val  = torch.exp(-0.5 * quad)                      # [h,w]

    # Per-pixel weight and color
    w = (a * val)[..., None]                           # [h,w,1]
    rgb_unit = torch.stack((rc, gc, bc)) / 255.0       # [3]
    C_add = w * rgb_unit                               # [h,w,3]

    # Accumulate
    C_num[y0:y1+1, x0:x1+1, :] += C_add
    W_sum[y0:y1+1, x0:x1+1, :] += w


@torch.no_grad()
def render_splats_rgb(
    genome: torch.Tensor,  # [N,9] or [9]
    H: int,
    W: int,
    *,
    k_sigma: float = 3.0,
    device: torch.device | None = None,
    background: tuple[float, float, float] | None = (0.0, 0.0, 0.0),
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Order-independent weighted-average renderer (no Beer–Lambert).

    Accumulate:
        C_num = Σ_i ( w_i * color_i )      # [H,W,3]
        W_sum = Σ_i ( w_i )                # [H,W,1]
      where w_i = alpha_i * N_i

    Final display (with optional background color in [0,1]):
        if W_sum > 0:    rgb = C_num / (W_sum + eps)
        else:            rgb = background

    Returns: rgb in [0,1], shape [H,W,3] on CPU.
    """
    dev = device or _DEV
    C_num = torch.zeros((H, W, 3), device=dev, dtype=torch.float32)
    W_sum = torch.zeros((H, W, 1), device=dev, dtype=torch.float32)

    if genome.ndim == 1:
        genome = genome.unsqueeze(0)

    for i in range(genome.shape[0]):
        _render_single_into_avg(C_num, W_sum, genome[i], H, W, k_sigma, dev)

    # Normalize
    rgb = C_num / (W_sum + eps)

    # Optional background where there is no coverage
    if background is not None:
        bg = torch.tensor(background, device=dev, dtype=torch.float32).view(1, 1, 3)
        mask = (W_sum <= eps)  # [H,W,1] boolean
        rgb = torch.where(mask.expand_as(rgb), bg.expand_as(rgb), rgb)

    return rgb.clamp(0.0, 1.0).cpu()
