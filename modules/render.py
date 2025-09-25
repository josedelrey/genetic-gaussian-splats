# src/ggs/render.py
from __future__ import annotations
import torch

# Prefer GPU if available; falls back to CPU.
_DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

__all__ = [
    "render_splats_rgb",
    "_render_single_into",    # kept public for your step-by-step tests
    "_DEV",
]

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def _build_L_from_logs(a_log: torch.Tensor, b_log: torch.Tensor, c_raw: torch.Tensor, dev) -> torch.Tensor:
    """
    Construct lower-triangular Cholesky L in *pixel units* from
      a_log, b_log (diagonal logs) and c_raw (signed shear):
        L = [[exp(a_log), 0],
             [c_raw,      exp(b_log)]]
    """
    l11 = torch.exp(a_log)
    l22 = torch.exp(b_log)
    # tiny floors to avoid degeneracy
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


# ------------------------------------------------------------
# Core: render one splat into premultiplied buffers
# ------------------------------------------------------------

@torch.no_grad()
def _render_single_into(
    img_C: torch.Tensor,  # [H,W,3], premultiplied color buffer (in-place)
    img_A: torch.Tensor,  # [H,W,1], alpha buffer (in-place)
    indiv: torch.Tensor,  # [9] = [x, y, a_log, b_log, c_raw, r, g, b, alpha]
    H: int,
    W: int,
    k_sigma: float,
    dev: torch.device,
) -> None:
    """
    OVER-composite a single Gaussian splat into (img_C, img_A), both premultiplied.

    Genome row layout:
      [x, y, a_log, b_log, c_raw, r, g, b, alpha]
        - x, y in [0,1] (normalized coords, origin top-left)
        - a_log, b_log are logs of *pixel* scales (diagonal of L)
        - c_raw is signed shear in pixels (off-diagonal)
        - r, g, b in [0,255]
        - alpha in [0,1]
    """
    x, y, a_log, b_log, c_raw, rc, gc, bc, a = indiv.to(dev, dtype=torch.float32)

    # Clamp only display ranges
    a  = torch.clamp(a,  0.0, 1.0)
    rc = torch.clamp(rc, 0.0, 255.0)
    gc = torch.clamp(gc, 0.0, 255.0)
    bc = torch.clamp(bc, 0.0, 255.0)

    # Center in pixels
    cx = torch.clamp(x, 0.0, 1.0) * (W - 1)
    cy = torch.clamp(y, 0.0, 1.0) * (H - 1)

    # Geometry from logs
    L = _build_L_from_logs(a_log, b_log, c_raw, dev)
    hx, hy = _aabb_from_L(L, k_sigma)

    # Crop (inclusive indices)
    x0 = int(torch.clamp(cx - hx, 0, W - 1).item())
    x1 = int(torch.clamp(cx + hx, 0, W - 1).item())
    y0 = int(torch.clamp(cy - hy, 0, H - 1).item())
    y1 = int(torch.clamp(cy + hy, 0, H - 1).item())
    if x1 <= x0 or y1 <= y0:
        return  # off-screen / degenerate

    # Local pixel grid
    xs = torch.arange(x0, x1 + 1, device=dev, dtype=torch.float32)
    ys = torch.arange(y0, y1 + 1, device=dev, dtype=torch.float32)
    X, Y = torch.meshgrid(xs, ys, indexing='xy')
    qx = X - cx
    qy = Y - cy

    # Quadratic form and Gaussian value
    l11 = L[0, 0]
    l21 = L[1, 0]
    l22 = L[1, 1]
    quad = _quadform_via_tri(qx, qy, l11, l21, l22)  # [h,w]
    val = torch.exp(-0.5 * quad)                      # [h,w]

    # Premultiplied contribution
    alpha_pix = (a * val)[..., None]                  # [h,w,1]
    rgb = torch.stack((rc, gc, bc)) / 255.0           # [3]
    C_add = alpha_pix * rgb                           # [h,w,3]

    # OVER-composite into the big buffers
    A_in = img_A[y0:y1+1, x0:x1+1, :]                 # [h,w,1]
    one_minus = 1.0 - A_in
    img_C[y0:y1+1, x0:x1+1, :] = img_C[y0:y1+1, x0:x1+1, :] + C_add * one_minus
    img_A[y0:y1+1, x0:x1+1, :] = A_in + alpha_pix * one_minus


# ------------------------------------------------------------
# Public: render N splats and return straight RGB [0,1]
# ------------------------------------------------------------

@torch.no_grad()
def render_splats_rgb(genome, H, W, *, k_sigma=3.0, device=None, unpremultiply=False, bg=None):
    dev = device or _DEV
    img_C = torch.zeros((H, W, 3), device=dev, dtype=torch.float32)
    img_A = torch.zeros((H, W, 1), device=dev, dtype=torch.float32)

    if genome.ndim == 1:
        genome = genome.unsqueeze(0)

    for i in range(genome.shape[0]):
        _render_single_into(img_C, img_A, genome[i], H, W, k_sigma, dev)

    if bg is not None:
        # composite over background (bg can be [3] or [H,W,3])
        if isinstance(bg, torch.Tensor):
            B = bg.to(dev).reshape(1, 1, 3).expand(H, W, 3)
        else:
            # tuple like (1,1,1) for white
            B = torch.tensor(bg, device=dev, dtype=torch.float32).reshape(1,1,3).expand(H,W,3)
        rgb = img_C + (1.0 - img_A) * B
    elif unpremultiply:
        eps = 1e-8
        rgb = torch.where(img_A > eps, img_C / img_A, img_C)
    else:
        rgb = img_C  # best for preview on black

    return rgb.clamp(0, 1).cpu()

