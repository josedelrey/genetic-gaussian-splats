from __future__ import annotations
import torch


_DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

__all__ = [
    "render_splats_rgb",           # order-dependent OVER compositing (no alpha)
    "_render_single_into_over",
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
def _render_single_into_over(
    C_img: torch.Tensor,   # [H,W,3] in-place image (0..1)
    indiv: torch.Tensor,   # [8] = [x, y, a_log, b_log, c_raw, r, g, b]
    H: int,
    W: int,
    k_sigma: float,
    dev: torch.device,
) -> None:
    """
    Order-dependent OVER compositing per splat (no alpha):
      f = Gaussian(x) in [0,1]
      C <- (1 - f) * C + f * color
    """
    x, y, a_log, b_log, c_raw, rc, gc, bc = indiv.to(dev, dtype=torch.float32)

    # Clamp colors
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

    # Gaussian weight in Σ metric
    l11 = L[0, 0]; l21 = L[1, 0]; l22 = L[1, 1]
    quad = _quadform_via_tri(qx, qy, l11, l21, l22)   # [h,w]
    val  = torch.exp(-0.5 * quad)                     # [h,w]

    # Blend factor and color
    f = val[..., None]                                # [h,w,1], no alpha
    color = torch.stack((rc, gc, bc)) / 255.0         # [3]

    # OVER compositing: C = (1 - f) * C + f * color
    region = C_img[y0:y1+1, x0:x1+1, :]
    C_img[y0:y1+1, x0:x1+1, :] = region * (1.0 - f) + f * color


@torch.no_grad()
def render_splats_rgb(
    genome: torch.Tensor,  # [N,8] or [8]
    H: int,
    W: int,
    *,
    k_sigma: float = 3.0,
    device: torch.device | None = None,
    background: tuple[float, float, float] = (1.0, 1.0, 1.0),  # white
) -> torch.Tensor:
    """
    Order-dependent OVER renderer (shader-style, no alpha):
        For each splat (in given order):
            f = Gaussian
            C <- (1 - f) * C + f * color
    Returns: rgb in [0,1], shape [H,W,3] on CPU.
    """
    dev = device or _DEV

    # Init image to background
    C_img = torch.tensor(background, device=dev, dtype=torch.float32).view(1, 1, 3)
    C_img = C_img.expand(H, W, 3).clone()

    if genome.ndim == 1:
        genome = genome.unsqueeze(0)

    for i in range(genome.shape[0]):
        _render_single_into_over(C_img, genome[i], H, W, k_sigma, dev)

    return C_img.clamp(0.0, 1.0).cpu()
