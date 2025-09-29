# modules/render_triton_batch.py
from __future__ import annotations
import torch, triton, triton.language as tl

_DEV = 'cuda'
__all__ = ["render_splats_rgb_triton", "_DEV"]

# ============================================================
# Preprocess: normalize coords, build Σ^{-1} elements, AABBs
# ============================================================
@torch.no_grad()
def _preprocess_genome(genome: torch.Tensor, H: int, W: int, k_sigma: float, device: torch.device):
    if genome.ndim == 1:
        genome = genome.unsqueeze(0)
    g = genome.to(device=device, dtype=torch.float32)

    maxx = float(W - 1); maxy = float(H - 1)
    cx = (g[:, 0].clamp(0.0, 1.0) * maxx)  # centers in pixels
    cy = (g[:, 1].clamp(0.0, 1.0) * maxy)

    # Cholesky params -> precision terms for quad form
    l11 = g[:, 2].exp().clamp_min(1e-6)
    l22 = g[:, 3].exp().clamp_min(1e-6)
    l21 = g[:, 4]

    # k-sigma AABB half-sizes (cheap, slightly conservative)
    hx = (k_sigma * l11.abs()).clamp_min(1.0)
    hy = (k_sigma * (l21.abs() + l22.abs())).clamp_min(1.0)

    x0 = (cx - hx).clamp(0, maxx).floor().to(torch.int32)
    x1 = (cx + hx).clamp(0, maxx).ceil().to(torch.int32)
    y0 = (cy - hy).clamp(0, maxy).floor().to(torch.int32)
    y1 = (cy + hy).clamp(0, maxy).ceil().to(torch.int32)

    inv_l11 = 1.0 / l11
    inv_l22 = 1.0 / l22
    inv_l21 = -l21 * (inv_l11 * inv_l22)

    sxx = inv_l11 * inv_l11 + inv_l21 * inv_l21
    sxy = inv_l21 * inv_l22
    syy = inv_l22 * inv_l22

    rc = g[:, 5].clamp(0.0, 255.0) / 255.0
    gc = g[:, 6].clamp(0.0, 255.0) / 255.0
    bc = g[:, 7].clamp(0.0, 255.0) / 255.0
    a  = g[:, 8].clamp(0.0, 255.0) / 255.0

    return {"cx":cx, "cy":cy, "sxx":sxx, "sxy":sxy, "syy":syy,
            "rc":rc, "gc":gc, "bc":bc, "a":a,
            "x0":x0, "x1":x1, "y0":y0, "y1":y1}

# ============================================================
# Fully GPU-vectorized binning (no Python loops, no CPU)
# Preserves OVER order within each tile by stable sorting
# first by (batch,tile), then by splat index.
# ============================================================
@torch.no_grad()
def _gpu_bin_splats_to_tiles(x0, x1, y0, y1, B:int, N:int, H:int, W:int, tile:int):
    device = x0.device
    nTX = (W + tile - 1) // tile
    nTY = (H + tile - 1) // tile
    ntiles = nTX * nTY
    S = B * N  # total splats

    # Tile ranges per splat
    tx0 = torch.div(x0, tile, rounding_mode='floor').clamp(0, nTX - 1)
    ty0 = torch.div(y0, tile, rounding_mode='floor').clamp(0, nTY - 1)
    tx1 = torch.div(x1, tile, rounding_mode='floor').clamp(0, nTX - 1)
    ty1 = torch.div(y1, tile, rounding_mode='floor').clamp(0, nTY - 1)

    nx = (tx1 - tx0 + 1).clamp_min(0)  # tiles covered in X
    ny = (ty1 - ty0 + 1).clamp_min(0)  # tiles covered in Y
    counts = (nx * ny)                 # tiles per splat

    valid_mask = counts > 0
    if not torch.any(valid_mask):
        flat_idx = torch.empty((0,), device=device, dtype=torch.int32)
        tile_cnt = torch.zeros((B * ntiles,), device=device, dtype=torch.int32)
        tile_off = torch.zeros_like(tile_cnt)
        return flat_idx, tile_off, tile_cnt, nTX, nTY, ntiles

    # Work only on valid splats
    valid_ids = torch.nonzero(valid_mask, as_tuple=False).flatten()  # [S_valid]
    counts_v = counts[valid_ids]
    tx0_v = tx0[valid_ids]; ty0_v = ty0[valid_ids]
    nx_v = nx[valid_ids]

    # Build expanded mapping for each valid splat
    total = counts_v.sum()
    rel = torch.arange(total, device=device, dtype=torch.int32)
    starts = torch.cumsum(counts_v, dim=0, dtype=torch.int32) - counts_v

    splat_ids_expanded = torch.repeat_interleave(valid_ids.to(torch.int32), counts_v)
    starts_per_expanded = torch.repeat_interleave(starts, counts_v)
    k = rel - starts_per_expanded  # 0..counts_v[i]-1 per splat

    nx_exp = torch.repeat_interleave(nx_v.to(torch.int32), counts_v)
    tx0_exp = torch.repeat_interleave(tx0_v.to(torch.int32), counts_v)
    ty0_exp = torch.repeat_interleave(ty0_v.to(torch.int32), counts_v)

    dx = torch.remainder(k, nx_exp)
    dy = torch.div(k, nx_exp, rounding_mode='floor')

    tx = tx0_exp + dx
    ty = ty0_exp + dy

    # Compute batch for each splat
    b_exp = torch.div(splat_ids_expanded, N, rounding_mode='floor')
    tile_local = ty * nTX + tx                      # 0..(ntiles-1)
    tile_global = b_exp * ntiles + tile_local       # 0..(B*ntiles-1)

    # Stable grouping by (tile_global, splat_idx)
    S64 = torch.tensor(int(S) + 1, device=device, dtype=torch.int64)
    sort_key = tile_global.to(torch.int64) * S64 + splat_ids_expanded.to(torch.int64)
    order = torch.argsort(sort_key)

    flat_idx = splat_ids_expanded[order].to(torch.int32)
    tile_global_sorted = tile_global[order]

    # Per-tile counts and offsets for all B*ntiles (including empty)
    tile_cnt = torch.bincount(tile_global_sorted.to(torch.int64),
                              minlength=B * ntiles).to(torch.int32)
    tile_off = (torch.cumsum(tile_cnt, dim=0, dtype=torch.int64) - tile_cnt.to(torch.int64)).to(torch.int32)

    return flat_idx, tile_off, tile_cnt, nTX, nTY, ntiles

# ============================================================
# Triton kernel: OVER compositing, one program per (batch,tile)
# ============================================================
@triton.jit
def _render_tile_over_kernel(
    C_ptr, H, W, stride_b, stride_h, stride_w,   # [B,H,W,3], channels-last contiguous
    cx_ptr, cy_ptr, sxx_ptr, sxy_ptr, syy_ptr,
    rc_ptr, gc_ptr, bc_ptr, a_ptr,
    x0_ptr, x1_ptr, y0_ptr, y1_ptr,
    flat_idx_ptr, tile_off_ptr, tile_cnt_ptr,
    nTX, ntiles,
    TILE_W: tl.constexpr, TILE_H: tl.constexpr,
):
    pid = tl.program_id(0)      # 0..(B*ntiles-1)
    b = pid // ntiles           # batch id
    t = pid % ntiles            # tile id inside this batch

    tx = t % nTX
    ty = t // nTX

    tile_x0 = tx * TILE_W
    tile_y0 = ty * TILE_H

    off_x = tl.arange(0, TILE_W)
    off_y = tl.arange(0, TILE_H)
    X = off_x[None, :] + tile_x0
    Y = off_y[:, None] + tile_y0

    in_x = X < W
    in_y = Y < H
    pix_mask = in_x & in_y

    # Canvas addressing (channels-last [B,H,W,3])
    base_b = b * stride_b
    base_addr = base_b + (Y * stride_h + X * stride_w)
    addr_r = base_addr + 0
    addr_g = base_addr + 1
    addr_b_ = base_addr + 2

    Cr = tl.load(C_ptr + addr_r, mask=pix_mask, other=0.0)
    Cg = tl.load(C_ptr + addr_g, mask=pix_mask, other=0.0)
    Cb = tl.load(C_ptr + addr_b_, mask=pix_mask, other=0.0)

    offs = tl.load(tile_off_ptr + (b * ntiles + t))
    cnt  = tl.load(tile_cnt_ptr + (b * ntiles + t))

    Xf = X.to(tl.float32); Yf = Y.to(tl.float32)

    k = 0
    while k < cnt:
        idx = tl.load(flat_idx_ptr + (offs + k))
        k += 1

        sx0 = tl.load(x0_ptr + idx); sx1 = tl.load(x1_ptr + idx)
        sy0 = tl.load(y0_ptr + idx); sy1 = tl.load(y1_ptr + idx)

        # quick AABB mask inside this tile
        in_x_aabb = (X >= sx0) & (X <= sx1)
        in_y_aabb = (Y >= sy0) & (Y <= sy1)
        m = pix_mask & in_x_aabb & in_y_aabb

        cx  = tl.load(cx_ptr  + idx)
        cy  = tl.load(cy_ptr  + idx)
        sxx = tl.load(sxx_ptr + idx)
        sxy = tl.load(sxy_ptr + idx)
        syy = tl.load(syy_ptr + idx)
        rc  = tl.load(rc_ptr  + idx)
        gc  = tl.load(gc_ptr  + idx)
        bc  = tl.load(bc_ptr  + idx)
        a   = tl.load(a_ptr   + idx)

        qx = Xf - cx
        qy = Yf - cy
        quad = sxx * (qx * qx) + 2.0 * sxy * (qx * qy) + syy * (qy * qy)
        f = tl.exp(-0.5 * quad) * a

        Cr = tl.where(m, (1.0 - f) * Cr + f * rc, Cr)
        Cg = tl.where(m, (1.0 - f) * Cg + f * gc, Cg)
        Cb = tl.where(m, (1.0 - f) * Cb + f * bc, Cb)

    tl.store(C_ptr + addr_r, Cr, mask=pix_mask)
    tl.store(C_ptr + addr_g, Cg, mask=pix_mask)
    tl.store(C_ptr + addr_b_, Cb, mask=pix_mask)

# ============================================================
# Public API (always GPU binning)
# ============================================================
@torch.no_grad()
def render_splats_rgb_triton(
    genomes: torch.Tensor,   # [B,N,9] or [N,9]
    H: int, W: int, *,
    k_sigma: float = 3.0,
    device: torch.device | str | None = None,
    background=(1.0, 1.0, 1.0),
    tile: int = 64,
    num_warps: int = 8,
    num_stages: int = 3,
    use_fp16_canvas: bool = False
) -> torch.Tensor:          # -> [B,H,W,3] on GPU
    dev = device or _DEV
    dev = torch.device(dev) if not isinstance(dev, torch.device) else dev
    assert dev.type == "cuda", "This renderer requires a CUDA device."

    assert genomes.ndim in (2, 3), f"genomes must be [B,N,9] or [N,9], got {genomes.shape}"
    if genomes.ndim == 2:
        genomes = genomes.unsqueeze(0)
    B, N, C = genomes.shape
    assert C >= 9, "expected at least 9 genome cols"

    # Preprocess per batch, then concat along splat-dimension
    parts = [_preprocess_genome(genomes[b], H, W, k_sigma, dev) for b in range(B)]
    cat = {k: torch.cat([p[k] for p in parts], dim=0) for k in parts[0].keys()}

    # Always GPU binning
    flat_idx, tile_off, tile_cnt, nTX, nTY, ntiles = _gpu_bin_splats_to_tiles(
        cat["x0"], cat["x1"], cat["y0"], cat["y1"], B, N, H, W, tile
    )

    # Allocate canvas [B,H,W,3] channels-last
    dtype = torch.float16 if use_fp16_canvas else torch.float32
    Cimg = torch.empty((B, H, W, 3), device=dev, dtype=dtype).contiguous()
    Cimg[:] = torch.as_tensor(background, device=dev, dtype=dtype)

    stride_b, stride_h, stride_w, _ = Cimg.stride()
    grid = (B * ntiles,)

    _render_tile_over_kernel[grid](
        Cimg, H, W, stride_b, stride_h, stride_w,
        cat["cx"], cat["cy"], cat["sxx"], cat["sxy"], cat["syy"],
        cat["rc"], cat["gc"], cat["bc"], cat["a"],
        cat["x0"], cat["x1"], cat["y0"], cat["y1"],
        flat_idx, tile_off, tile_cnt,
        nTX, ntiles,
        TILE_W=tile, TILE_H=tile,
        num_warps=num_warps, num_stages=num_stages,
    )
    return Cimg.clamp_(0.0, 1.0).to(torch.float32)
