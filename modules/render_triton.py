from __future__ import annotations
import torch
import triton
import triton.language as tl

_DEV = 'cuda'
__all__ = ["render_splats_rgb_triton", "_DEV"]


# ---------------------------
# Host-side preprocessing
# ---------------------------
@torch.no_grad()
def _preprocess_genome(genome: torch.Tensor, H: int, W: int, k_sigma: float, device: torch.device):
    """
    Input genome: [N,9] = [x,y,a_log,b_log,c_raw,r,g,b,a] with colors/alpha in 0..255.
    Returns Σ^{-1} elements for quad-form, plus AABBs and normalized colors.
    NOTE: NO in-place writes on the input (avoids cumulative posterization).
    """
    if genome.ndim == 1:
        genome = genome.unsqueeze(0)

    # IMPORTANT: don't rely on .to() to make a distinct copy; keep it view-safe
    g = genome.to(device=device, dtype=torch.float32)  # may share storage; treat as read-only

    # centers in pixels (out-of-place)
    maxx = float(W - 1)
    maxy = float(H - 1)
    cx = (g[:, 0].clamp(0.0, 1.0) * maxx)
    cy = (g[:, 1].clamp(0.0, 1.0) * maxy)

    # Cholesky L = [[l11, 0],[l21, l22]] (out-of-place)
    l11 = g[:, 2].exp().clamp_min(1e-6)
    l22 = g[:, 3].exp().clamp_min(1e-6)
    l21 = g[:, 4]

    # AABB half-extents (conservative k-sigma box)
    hx = (k_sigma * l11.abs()).clamp_min(1.0)
    hy = (k_sigma * (l21.abs() + l22.abs())).clamp_min(1.0)

    # integer AABB (inclusive)
    x0 = (cx - hx).clamp(0, maxx).floor().to(torch.int32)
    x1 = (cx + hx).clamp(0, maxx).ceil().to(torch.int32)
    y0 = (cy - hy).clamp(0, maxy).floor().to(torch.int32)
    y1 = (cy + hy).clamp(0, maxy).ceil().to(torch.int32)

    # Σ^{-1} from L^{-1} (A = [[a,0],[b,c]] with a=1/l11, b=-l21/(l11*l22), c=1/l22)
    inv_l11 = 1.0 / l11
    inv_l22 = 1.0 / l22
    inv_l21 = -l21 * (inv_l11 * inv_l22)
    sxx = inv_l11 * inv_l11 + inv_l21 * inv_l21
    sxy = inv_l21 * inv_l22
    syy = inv_l22 * inv_l22

    # Colors/alpha normalized to 0..1 (out-of-place; no write to g)
    rc = g[:, 5].clamp(0.0, 255.0) / 255.0
    gc = g[:, 6].clamp(0.0, 255.0) / 255.0
    bc = g[:, 7].clamp(0.0, 255.0) / 255.0
    a  = g[:, 8].clamp(0.0, 255.0) / 255.0

    return {
        "cx": cx, "cy": cy,
        "sxx": sxx, "sxy": sxy, "syy": syy,
        "rc": rc, "gc": gc, "bc": bc, "a": a,
        "x0": x0, "x1": x1, "y0": y0, "y1": y1,
    }



@torch.no_grad()
def _bin_splats_to_tiles(x0, x1, y0, y1, H: int, W: int, tile: int):
    """Build compact tile → indices mapping."""
    nTX = (W + tile - 1) // tile
    nTY = (H + tile - 1) // tile
    ntiles = nTX * nTY
    bins = [[] for _ in range(ntiles)]

    X0 = x0.cpu().tolist(); X1 = x1.cpu().tolist()
    Y0 = y0.cpu().tolist(); Y1 = y1.cpu().tolist()

    for i in range(len(X0)):
        tx0 = X0[i] // tile
        tx1 = X1[i] // tile
        ty0 = Y0[i] // tile
        ty1 = Y1[i] // tile
        for ty in range(ty0, ty1 + 1):
            for tx in range(tx0, tx1 + 1):
                bins[ty * nTX + tx].append(i)

    counts = [len(b) for b in bins]
    offsets, flat, running = [], [], 0
    for b in bins:
        offsets.append(running)
        flat.extend(b)
        running += len(b)

    device = x0.device
    flat_indices = (torch.tensor(flat, device=device, dtype=torch.int32)
                    if flat else torch.empty((0,), device=device, dtype=torch.int32))
    tile_offset  = torch.tensor(offsets, device=device, dtype=torch.int32)
    tile_count   = torch.tensor(counts,  device=device, dtype=torch.int32)
    return flat_indices, tile_offset, tile_count, nTX, nTY


# ---------------------------
# Triton kernel (quad-form via Σ^{-1}; fp32 coords; OVER compositing)
# ---------------------------
@triton.jit
def _render_tile_over_kernel(
    C_ptr, H, W, stride_h, stride_w,
    # per-splat arrays
    cx_ptr, cy_ptr, sxx_ptr, sxy_ptr, syy_ptr,
    rc_ptr, gc_ptr, bc_ptr, a_ptr,
    x0_ptr, x1_ptr, y0_ptr, y1_ptr,
    # tile indirection
    flat_idx_ptr, tile_offset_ptr, tile_count_ptr, nTX,
    # compile-time tile size
    TILE_W: tl.constexpr, TILE_H: tl.constexpr,
):
    tid = tl.program_id(0)
    tx = tid % nTX
    ty = tid // nTX
    tile_x0 = tx * TILE_W
    tile_y0 = ty * TILE_H

    off_x = tl.arange(0, TILE_W)
    off_y = tl.arange(0, TILE_H)
    X = off_x[None, :] + tile_x0
    Y = off_y[:, None] + tile_y0

    # explicit fp32 to avoid integer arithmetic quirks
    Xf = X.to(tl.float32)
    Yf = Y.to(tl.float32)

    in_x = X < W
    in_y = Y < H
    pix_mask = in_x & in_y

    base_addr = Y * stride_h + X * stride_w
    addr_r = base_addr + 0
    addr_g = base_addr + 1
    addr_b = base_addr + 2

    Cr = tl.load(C_ptr + addr_r, mask=pix_mask, other=0.0)
    Cg = tl.load(C_ptr + addr_g, mask=pix_mask, other=0.0)
    Cb = tl.load(C_ptr + addr_b, mask=pix_mask, other=0.0)

    offs = tl.load(tile_offset_ptr + tid)
    cnt  = tl.load(tile_count_ptr  + tid)

    k = 0
    while k < cnt:
        idx = tl.load(flat_idx_ptr + (offs + k))
        k += 1

        # splat AABB (inclusive)
        sx0 = tl.load(x0_ptr + idx)
        sx1 = tl.load(x1_ptr + idx)
        sy0 = tl.load(y0_ptr + idx)
        sy1 = tl.load(y1_ptr + idx)

        # scalar 'active' if tile overlaps splat
        tx0 = tile_x0
        ty0 = tile_y0
        tx1 = tile_x0 + TILE_W - 1
        ty1 = tile_y0 + TILE_H - 1
        active = ~((sx1 < tx0) | (sx0 > tx1) | (sy1 < ty0) | (sy0 > ty1))

        cx  = tl.load(cx_ptr  + idx)
        cy  = tl.load(cy_ptr  + idx)
        sxx = tl.load(sxx_ptr + idx)
        sxy = tl.load(sxy_ptr + idx)
        syy = tl.load(syy_ptr + idx)
        rc  = tl.load(rc_ptr  + idx)
        gc  = tl.load(gc_ptr  + idx)
        bc  = tl.load(bc_ptr  + idx)
        a   = tl.load(a_ptr   + idx)

        in_x_aabb = (X >= sx0) & (X <= sx1)
        in_y_aabb = (Y >= sy0) & (Y <= sy1)
        m = active & pix_mask & in_x_aabb & in_y_aabb

        # quad-form: q^T Σ^{-1} q (parity with CPU after Σ^{-1} fix)
        qx = Xf - cx
        qy = Yf - cy
        quad = sxx * (qx * qx) + 2.0 * sxy * (qx * qy) + syy * (qy * qy)

        # Effective alpha (no cutoffs/gamma; identical to CPU)
        f = tl.exp(-0.5 * quad) * a

        # Porter–Duff OVER
        Cr = tl.where(m, (1.0 - f) * Cr + f * rc, Cr)
        Cg = tl.where(m, (1.0 - f) * Cg + f * gc, Cg)
        Cb = tl.where(m, (1.0 - f) * Cb + f * bc, Cb)

    tl.store(C_ptr + addr_r, Cr, mask=pix_mask)
    tl.store(C_ptr + addr_g, Cg, mask=pix_mask)
    tl.store(C_ptr + addr_b, Cb, mask=pix_mask)


# ---------------------------
# Public API
# ---------------------------
@torch.no_grad()
def render_splats_rgb_triton(
    genome: torch.Tensor,   # [N,9] (r,g,b,a in 0..255)
    H: int,
    W: int,
    *,
    k_sigma: float = 3.0,
    device: torch.device | str | None = None,
    background=(1.0, 1.0, 1.0),
    tile: int = 32,
) -> torch.Tensor:
    dev = device or _DEV
    dev = torch.device(dev) if not isinstance(dev, torch.device) else dev
    assert dev.type == "cuda", "render_splats_rgb_triton requires a CUDA device"

    P = _preprocess_genome(genome, H, W, k_sigma, dev)
    flat_idx, tile_off, tile_cnt, nTX, nTY = _bin_splats_to_tiles(P["x0"], P["x1"], P["y0"], P["y1"], H, W, tile)

    # channels-last contiguous (H,W,3)
    C = torch.empty((H, W, 3), device=dev, dtype=torch.float32).contiguous()
    C[:] = torch.tensor(background, device=dev, dtype=torch.float32)

    stride_h = C.stride(0)
    stride_w = C.stride(1)
    grid = (nTX * nTY,)

    _render_tile_over_kernel[grid](
        C, H, W, stride_h, stride_w,
        P["cx"], P["cy"], P["sxx"], P["sxy"], P["syy"],
        P["rc"], P["gc"], P["bc"], P["a"],
        P["x0"], P["x1"], P["y0"], P["y1"],
        flat_idx, tile_off, tile_cnt, nTX,
        TILE_W=tile, TILE_H=tile,
        num_warps=4, num_stages=2,
    )

    return C.clamp_(0.0, 1.0).cpu()
