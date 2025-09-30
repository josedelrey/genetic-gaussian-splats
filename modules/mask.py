import torch
import torch.nn.functional as F


@torch.no_grad()
def _rgb_to_luma(img_hw3: torch.Tensor) -> torch.Tensor:
    x = img_hw3
    if x.max() > 1.5: x = x / 255.0
    y = 0.2126 * x[...,0] + 0.7152 * x[...,1] + 0.0722 * x[...,2]
    return y.unsqueeze(0).unsqueeze(0).contiguous()


def _sobel_edges(y: torch.Tensor) -> torch.Tensor:
    kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=y.dtype, device=y.device).view(1,1,3,3)
    ky = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=y.dtype, device=y.device).view(1,1,3,3)
    gx = F.conv2d(y, kx, padding=1)
    gy = F.conv2d(y, ky, padding=1)
    return torch.sqrt(gx*gx + gy*gy + 1e-12)


def _local_variance(y: torch.Tensor, k:int=9) -> torch.Tensor:
    pad = k//2
    mean  = F.avg_pool2d(y, k, stride=1, padding=pad)
    mean2 = F.avg_pool2d(y*y, k, stride=1, padding=pad)
    return (mean2 - mean*mean).clamp_min(0)


@torch.no_grad()
def compute_importance_mask(
    target_hw3: torch.Tensor, H: int, W: int,
    edge_scales=(1,2,4),
    w_edge: float = 0.7,
    w_var: float  = 0.3,
    gamma: float  = 0.7,
    floor: float  = 0.15,
    smooth: int   = 0,
    strength: float = 1.0
) -> torch.Tensor:
    dev = target_hw3.device

    # Use correct resolution
    x = target_hw3
    if x.max() > 1.5: x = x / 255.0
    x4 = x.permute(2,0,1).unsqueeze(0)  # [1,3,H0,W0]
    x4 = F.interpolate(x4, size=(H,W), mode='bilinear', align_corners=False)
    y  = _rgb_to_luma(x4[0].permute(1,2,0))  # [1,1,H,W] on dev

    # Multi-scale edge magnitude
    edges = torch.zeros_like(y)
    for s in edge_scales:
        if s > 1:
            yd = F.avg_pool2d(y, kernel_size=s, stride=s)
            e  = _sobel_edges(yd)
            e  = F.interpolate(e, size=(H,W), mode='bilinear', align_corners=False)
        else:
            e  = _sobel_edges(y)
        edges = edges + e

    # Local variance
    var = _local_variance(y, k=9)

    # Robust normalize each cue to 0..1
    def _norm01(t: torch.Tensor) -> torch.Tensor:
        ql = torch.quantile(t.flatten(), 0.02)
        qh = torch.quantile(t.flatten(), 0.98)
        return ((t - ql) / (qh - ql + 1e-12)).clamp(0,1)

    E = _norm01(edges)
    V = _norm01(var)

    mask = (w_edge * E + w_var * V)
    mask = _norm01(mask)
    if smooth and smooth > 0:
        mask = F.avg_pool2d(mask, kernel_size=smooth, stride=1, padding=smooth//2)
        mask = _norm01(mask)

    mask = mask.pow(gamma)
    mask = (1.0 - floor) * mask + floor

    # Blend with ones to control global strength
    if strength < 1.0:
        mask = (1.0 - strength) * torch.ones_like(mask) + strength * mask

    return mask[0,0]  # [H,W] on dev
