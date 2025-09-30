from __future__ import annotations

import os, math
import numpy as np
import torch, random
import torch.nn.functional as F
from typing import Tuple, List
from PIL import Image
from tqdm.auto import tqdm

from modules.render_triton import render_splats_rgb_triton, _DEV as DEV
from modules.resize import choose_work_size, scale_genome_pixels_anisotropic
from modules.encode import genome_to_renderer  # single-sample; we wrap it batched


INPUT_DIR    = "imgs"
OUTPUT_DIR   = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SAVE_VIDEO   = True
FRAME_EVERY  = 5000

VIDEO_DIR    = os.path.join(OUTPUT_DIR, "video_frames")
if SAVE_VIDEO:
    os.makedirs(VIDEO_DIR, exist_ok=True)

H, W          = 128, 128
WORK_MAX_SIDE = 512
N_SPLATS      = 512
POP_SIZE      = 8
GENERATIONS   = 500000
TOUR_K        = 2
ELITE_K       = 4
CXPB          = 0.05

MUTPB = 0.01
PROTECT_BEST_ELITE = True

K_SIGMA       = 3.0
SEED          = 42

MIN_SCALE_SPLATS = 0.8            # px (lets you draw hairlines)
MAX_SCALE_SPLATS = 0.15           # → ~90 px at 256 for broad strokes

# xy: fraction of image size
# a_log, b_log: log()

# MUT_SIGMA_MAX = {"xy":0.1, "alog":0.5, "blog":0.5, "theta":0.3,
#                  "rgb":20.0, "alpha":20.0}
# MUT_SIGMA_MIN = {"xy":0.005,"alog":0.01, "blog":0.01, "theta":0.01,
#                  "rgb":1.0, "alpha":1.0}

MUT_SIGMA_MAX = {"xy":0.1, "alog":0.5, "blog":0.5, "theta":0.3,
                 "rgb":25.0, "alpha":25.0}
MUT_SIGMA_MIN = {"xy":0.01,"alog":0.05, "blog":0.05, "theta":0.025,
                 "rgb":5.0, "alpha":5.0}

SCHEDULE = "cosine"  # "linear", "cosine", "exp"

# Toggle strength of the importance mask blending (1.0 = full, 0.0 = plain MSE)
MASK_STRENGTH = 0.7
BOOST_ONLY    = False  # If True, use mask to only boost important areas instead of weighting all pixels

# Repro
if SEED is not None:
    random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

# ============================================================
# Importance mask (multi-scale edges + local variance)
# ============================================================
@torch.no_grad()
def _rgb_to_luma(img_hw3: torch.Tensor) -> torch.Tensor:
    """
    img_hw3: [H,W,3], 0..1 or 0..255, on any device
    returns: [1,1,H,W], 0..1 on same device
    """
    x = img_hw3
    if x.max() > 1.5: x = x / 255.0
    y = 0.2126 * x[...,0] + 0.7152 * x[...,1] + 0.0722 * x[...,2]
    return y.unsqueeze(0).unsqueeze(0).contiguous()

def _sobel_edges(y: torch.Tensor) -> torch.Tensor:
    """
    y: [1,1,H,W] → gradient magnitude [1,1,H,W]
    """
    kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=y.dtype, device=y.device).view(1,1,3,3)
    ky = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=y.dtype, device=y.device).view(1,1,3,3)
    gx = F.conv2d(y, kx, padding=1)
    gy = F.conv2d(y, ky, padding=1)
    return torch.sqrt(gx*gx + gy*gy + 1e-12)

def _local_variance(y: torch.Tensor, k:int=9) -> torch.Tensor:
    """
    y: [1,1,H,W] → local variance [1,1,H,W]
    """
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
    smooth: int   = 0,      # optional blur radius (0 disables)
    strength: float = 1.0   # 0..1: blend with ones to soften effect
) -> torch.Tensor:
    """
    Returns [H,W] weights in [0,1] on the same device as target.
    """
    dev = target_hw3.device

    # Ensure we're using the working resolution
    x = target_hw3
    if x.max() > 1.5: x = x / 255.0
    x4 = x.permute(2,0,1).unsqueeze(0)                              # [1,3,H0,W0]
    x4 = F.interpolate(x4, size=(H,W), mode='bilinear', align_corners=False)
    y  = _rgb_to_luma(x4[0].permute(1,2,0))                         # [1,1,H,W] on dev

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

    # Local variance (texture/contrast)
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
    mask = (1.0 - floor) * mask + floor       # non-zero everywhere

    # Blend with ones to control global strength
    if strength < 1.0:
        mask = (1.0 - strength) * torch.ones_like(mask) + strength * mask

    return mask[0,0]                           # [H,W] on dev

# ============================================================
# Annealing utils
# ============================================================
def _anneal_factor(gen, total, kind):
    g = max(0, min(gen, total))
    p = g / max(1, total)
    if kind == "cosine":
        raw = 0.5 * (1.0 + math.cos(math.pi * p))
    elif kind == "linear":
        raw = 1.0 - p
    elif kind == "exp":
        target = 0.2
        decay = target ** (1.0 / max(1, total))
        raw = decay ** g
    else:
        raw = 1.0 - p
    return max(0.0, raw)

def build_mut_sigma(gen:int, total_gens:int, kind:str):
    f = _anneal_factor(gen, total_gens, kind)
    return {k: MUT_SIGMA_MIN[k] + f * (MUT_SIGMA_MAX[k] - MUT_SIGMA_MIN[k]) for k in MUT_SIGMA_MAX.keys()}

# ============================================================
# Genome helpers (row: [x, y, a_log, b_log, theta, r, g, b, alpha255])
# ============================================================
@torch.no_grad()
def wrap_angle(theta: torch.Tensor) -> torch.Tensor:
    return (theta + np.pi) % (2*np.pi) - np.pi

def clamp_genome(ind: torch.Tensor) -> torch.Tensor:
    ind[:,0:2] = ind[:,0:2].clamp(0.0, 1.0)
    max_side = float(max(H, W))
    min_scale_log = torch.log(torch.tensor(MIN_SCALE_SPLATS, device=ind.device))
    max_scale_log = torch.log(torch.tensor(MAX_SCALE_SPLATS * max_side, device=ind.device))
    ind[:,2] = ind[:,2].clamp(min_scale_log, max_scale_log)
    ind[:,3] = ind[:,3].clamp(min_scale_log, max_scale_log)
    ind[:,4] = wrap_angle(ind[:,4])
    ind[:,5:9] = ind[:,5:9].clamp(0.0, 255.0)
    return ind

@torch.no_grad()
def new_population(batch_size:int, n_splats:int, H:int, W:int, device=DEV, dtype=torch.float32) -> torch.Tensor:
    """
    Returns [B,N,9] axes+theta genome directly, vectorized on device.
    Layout per row: [x, y, a_log, b_log, theta, r, g, b, alpha255]
    """
    B, N = batch_size, n_splats
    max_side = float(max(H, W))

    # xy in [0,1]
    xy = torch.empty(B, N, 2, device=device, dtype=dtype).uniform_(0.0, 1.0)

    # scales (sample in linear-sigma, then log)
    s_lo = float(MIN_SCALE_SPLATS)
    s_hi = float(MAX_SCALE_SPLATS * max_side)
    a = torch.empty(B, N, 1, device=device, dtype=dtype).uniform_(s_lo, s_hi).log()
    b = torch.empty(B, N, 1, device=device, dtype=dtype).uniform_(s_lo, s_hi).log()

    # theta
    theta = torch.empty(B, N, 1, device=device, dtype=dtype).uniform_(-math.pi, math.pi)

    # colors + alpha (uint8 range but kept float here)
    rgb   = torch.empty(B, N, 3, device=device, dtype=dtype).uniform_(0.0, 256.0)
    alpha = torch.empty(B, N, 1, device=device, dtype=dtype).uniform_(180.0, 256.0)

    G = torch.cat([xy, a, b, theta, rgb, alpha], dim=-1)  # [B,N,9]
    # Clamp once (angle wrap optional here; theta is already in-range)
    G[..., 0:2].clamp_(0.0, 1.0)
    G[..., 5:9].clamp_(0.0, 255.0)
    return G

def new_individual(n_splats=N_SPLATS, device=DEV) -> torch.Tensor:
    return new_population(1, n_splats, H, W, device=device)[0]

def duplicate_individual(ind:torch.Tensor) -> torch.Tensor:
    return ind.clone()

# ============================================================
# Batched conversion helper (no Python loops)
# Converts [B,N,8/9] axes+theta → [B,N,9] renderer layout.
# ============================================================
@torch.no_grad()
def genome_to_renderer_batched(G_axes: torch.Tensor) -> torch.Tensor:
    """
    G_axes: [B,N,C] with C in {8,9}; axes-angle format
    Returns: [B,N,9] = [x,y,a_log_eff,b_log_eff,c_raw_eff,r,g,b,alpha]
    """
    assert G_axes.ndim == 3, f"Expected [B,N,C], got {G_axes.shape}"
    B, N, C = G_axes.shape
    Gf = G_axes.reshape(B * N, C)

    R = genome_to_renderer(Gf)  # [B*N, 9 or >9]
    if R.shape[1] < 9:
        if C >= 9:
            a = Gf[:, 8:9]
        else:
            a = torch.full((R.shape[0], 1), 255.0, device=R.device, dtype=R.dtype)
        R = torch.cat([R, a], dim=1)
    elif R.shape[1] > 9:
        R = R[:, :9]

    R[:, 5:9].clamp_(0.0, 255.0)

    return R.reshape(B, N, 9)

# ============================================================
# Rendering helpers (use the SAME batched renderer)
# ============================================================
@torch.no_grad()
def render_axes_angle_to_img(ind_axes_angle: torch.Tensor, Hsnap:int, Wsnap:int) -> np.ndarray:
    G = ind_axes_angle.unsqueeze(0) if ind_axes_angle.ndim == 2 else ind_axes_angle  # [1,N,C]
    G9 = genome_to_renderer_batched(G)                                               # [1,N,9]
    img = render_splats_rgb_triton(G9, Hsnap, Wsnap, k_sigma=K_SIGMA, device=DEV, tile=32)[0]  # [H,W,3]
    img8 = (img.clamp(0,1).detach().cpu().numpy() * 255.0).astype("uint8")
    return img8

@torch.no_grad()
def save_frame_png(gen:int, ind_axes_angle: torch.Tensor, pad:int, prefix:str="frame"):
    if not SAVE_VIDEO: return
    img8 = render_axes_angle_to_img(ind_axes_angle, H, W)
    fname = f"{prefix}_{gen:0{pad}d}.png"
    Image.fromarray(img8).save(os.path.join(VIDEO_DIR, fname))

@torch.no_grad()
def prewarm_renderer(H:int, W:int, device=DEV):
    """Trigger CUDA context + Triton JIT once, on tiny shapes, and cache variants."""
    dummy = torch.tensor([[[0.5,0.5, math.log(2.0), math.log(2.0), 0.0, 128.0,128.0,128.0, 255.0]]],
                         device=device, dtype=torch.float32)  # [1,1,9]
    _ = render_splats_rgb_triton(dummy, min(8,H), min(8,W), k_sigma=K_SIGMA, device=device, tile=32)
    _ = render_splats_rgb_triton(dummy, min(8,H), min(8,W), k_sigma=K_SIGMA, device=device, tile=32)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

# ============================================================
# Fitness (batched) — weighted MSE with optional importance mask
# ============================================================
@torch.no_grad()
def fitness_many(pop_batch, target, tile: int = 32,
                 weight_mask: torch.Tensor | None = None,
                 boost_only: bool = False,  # <-- new flag
                 boost_beta: float = 1.0):  # weights in [1, 1+beta]
    G_axes = torch.stack(pop_batch, dim=0)          # [B,N,C]
    G9 = genome_to_renderer_batched(G_axes)         # [B,N,9]
    imgs = render_splats_rgb_triton(G9, H, W, k_sigma=K_SIGMA, device=DEV, tile=tile)  # [B,H,W,3]
    dif2 = (imgs - target.unsqueeze(0)) ** 2        # [B,H,W,3]

    if weight_mask is None:
        return dif2.mean(dim=(1,2,3))

    w = weight_mask.unsqueeze(0).unsqueeze(-1)      # [1,H,W,1]

    if boost_only:
        # normalize mask to [0,1] first (should already be), then shift to [1, 1+beta]
        # If your mask is already 0..1, you can skip the clamp.
        w_boost = 1.0 + boost_beta * w.clamp(0, 1)
        # Normalize by mean(w_boost) so scale ~ plain MSE
        num = (dif2 * w_boost).mean(dim=(1,2,3))    # average over pixels
        den = w_boost.mean(dim=(1,2,3)) + 1e-12
        return num / den
    else:
        # relative emphasis (your current behavior)
        num = (dif2 * w).sum(dim=(1,2,3))
        den = (w.sum(dim=(1,2,3)) + 1e-12)
        return num / den


@torch.no_grad()
def fitness_population(population: List[torch.Tensor], target: torch.Tensor,
                       tile: int = 32, chunk: int | None = None,
                       weight_mask: torch.Tensor | None = None) -> List[float]:
    """
    Evaluate the whole population in CUDA batches. If chunk is None, do it in one go.
    Returns a Python list of floats.
    """
    if chunk is None or chunk >= len(population):
        return fitness_many(population, target, tile=tile, weight_mask=weight_mask, boost_only=BOOST_ONLY).detach().cpu().tolist()
    out: List[float] = []
    for i in range(0, len(population), chunk):
        out.extend(fitness_many(population[i:i+chunk], target, tile=tile, weight_mask=weight_mask, boost_only=BOOST_ONLY).detach().cpu().tolist())
    return out

# ============================================================
# Selection, Crossover, Mutation
# ============================================================
def tournament_selection(pop, fits, k=TOUR_K):
    best_idx = None
    for _ in range(k):
        i = random.randrange(len(pop))
        if best_idx is None or fits[i] < fits[best_idx]:
            best_idx = i
    return duplicate_individual(pop[best_idx])

def crossover_uniform(a, b, p=0.5):
    m = (torch.rand((a.shape[0],1), device=a.device) < p)
    child1 = torch.where(m, a, b)
    child2 = torch.where(m, b, a)
    return child1, child2

def _ensure_one_true(mask: torch.Tensor) -> torch.Tensor:
    if not mask.any():
        flat = mask.view(-1)
        k = int(torch.randint(flat.numel(), (1,), device=mask.device).item())
        flat[k] = True
    return mask

def mutate_individual(ind, is_elite, gen, total_gens, schedule):
    SIG = build_mut_sigma(gen, total_gens, schedule)

    with torch.no_grad():
        N = ind.shape[0]

        # ------------------ masks ------------------
        m_xy = (torch.rand((N, 2), device=ind.device) < MUTPB)
        m_ab = (torch.rand((N, 2), device=ind.device) < MUTPB)
        m_t  = (torch.rand((N, 1), device=ind.device) < MUTPB)

        # Color: sample 1 flag for the whole RGB triplet, and 1 for alpha
        m_rgb_flag = (torch.rand((N, 1), device=ind.device) < MUTPB)
        m_a_flag   = (torch.rand((N, 1), device=ind.device) < MUTPB)

        m_color_pair = torch.cat([m_rgb_flag, m_a_flag], dim=1)
        m_color_pair = _ensure_one_true(m_color_pair)
        m_rgb_flag = m_color_pair[:, 0:1]
        m_a_flag   = m_color_pair[:, 1:2]

        m_rgba = torch.cat([m_rgb_flag.expand(-1, 3), m_a_flag], dim=1)

        m_xy = _ensure_one_true(m_xy)
        m_ab = _ensure_one_true(m_ab)
        m_t  = _ensure_one_true(m_t)

        # ------------------ numeric mutations ------------------
        ind[:, 0:2] += torch.randn_like(ind[:, 0:2]) * SIG["xy"] * m_xy.float()

        ind[:, 2:4] += (
            torch.randn_like(ind[:, 2:4])
            * torch.tensor([SIG["alog"], SIG["blog"]], device=ind.device, dtype=ind.dtype)
            * m_ab.float()
        )

        ind[:, 4:5] += torch.randn_like(ind[:, 4:5]) * SIG["theta"] * m_t.float()
        ind[:, 4] = wrap_angle(ind[:, 4])

        sig_rgba = torch.tensor([SIG["rgb"], SIG["rgb"], SIG["rgb"], SIG["alpha"]],
                                device=ind.device, dtype=ind.dtype)
        ind[:, 5:9] += torch.randn_like(ind[:, 5:9]) * sig_rgba * m_rgba.float()

        clamp_genome(ind)

        # ------------------ render-order swap (detail-preserving) ------------------
        if N >= 2:
            i = int(torch.randint(0, N - 1, (1,), device=ind.device).item())
            size = torch.exp(ind[:, 2]) * torch.exp(ind[:, 3])  # proxy: sigma_x * sigma_y
            later = torch.arange(i + 1, N, device=ind.device)
            if later.numel() > 0:
                bigger_mask = size[later] > size[i]
                if bigger_mask.any():
                    candidates = later[bigger_mask]
                    j = int(candidates[torch.randint(0, candidates.numel(), (1,), device=ind.device)].item())
                    tmp = ind[i].clone()
                    ind[i] = ind[j]
                    ind[j] = tmp

    return ind

# ============================================================
# Main GA loop (with tqdm + frame export)
# ============================================================
@torch.no_grad()
def genetic_approx(target_img_uint8: torch.Tensor) -> Tuple[torch.Tensor, float]:
    # Prepare target at working resolution on CPU first
    t = target_img_uint8.to(torch.float32)
    if t.max() > 1.5: t = t / 255.0
    if t.shape[0] != H or t.shape[1] != W:
        tBCHW = t.permute(2,0,1).unsqueeze(0)
        t = F.interpolate(tBCHW, size=(H,W), mode='bilinear', align_corners=False)[0].permute(1,2,0)
    t = t.contiguous()                   # [H,W,3] on CPU, 0..1

    # Compute importance mask ONCE at working resolution, then move to GPU
    imp_mask = compute_importance_mask(
        t, H, W,
        edge_scales=(1,2,4),
        w_edge=0.7, w_var=0.3,
        gamma=0.7, floor=0.15,
        smooth=3,                    # small smoothing to avoid halo-chasing
        strength=MASK_STRENGTH
    ).to(DEV)                         # [H,W]

    # Move target to GPU
    target = t.to(DEV)

    # ---- Pre-warm CUDA + Triton JIT (fast tiny render) ----
    prewarm_renderer(H, W, device=DEV)

    # ---- Init population on CUDA (vectorized, single call) ----
    pop_tensor = new_population(POP_SIZE, N_SPLATS, H, W, device=DEV)  # [P,N,9]
    population = [pop_tensor[i] for i in range(POP_SIZE)]

    # ---- Fitness (batched) ----
    INIT_CHUNK = None  # set to e.g. 32 if VRAM limited
    fitnesses  = fitness_population(population, target, tile=32, chunk=INIT_CHUNK, weight_mask=imp_mask)

    best_idx = min(range(POP_SIZE), key=lambda i: fitnesses[i])
    best_ind = duplicate_individual(population[best_idx])
    best_fit = fitnesses[best_idx]
    no_improve = 0

    pad = len(str(GENERATIONS))
    if SAVE_VIDEO and (0 % max(1, FRAME_EVERY) == 0):
        save_frame_png(0, best_ind, pad, prefix="ga")

    pbar = tqdm(range(1, GENERATIONS+1), desc="GA generations", leave=True)
    try:
        for gen in pbar:
            parents = []
            while len(parents) < POP_SIZE:
                parents.append(tournament_selection(population, fitnesses, k=TOUR_K))

            random.shuffle(parents)

            offspring = []
            for i in range(0, POP_SIZE, 2):
                a = parents[i]
                b = parents[(i+1) % POP_SIZE]
                if random.random() < CXPB:
                    c1, c2 = crossover_uniform(a, b)
                else:
                    c1, c2 = duplicate_individual(a), duplicate_individual(b)
                offspring.append(mutate_individual(c1, is_elite=False, gen=gen, total_gens=GENERATIONS, schedule=SCHEDULE))
                if len(offspring) < POP_SIZE:
                    offspring.append(mutate_individual(c2, is_elite=False, gen=gen, total_gens=GENERATIONS, schedule=SCHEDULE))

            # Offspring fitness (batched)
            OFFSPRING_CHUNK = None
            off_fits = fitness_population(offspring, target, tile=32, chunk=OFFSPRING_CHUNK, weight_mask=imp_mask)

            # Elites
            elite_k = max(1, ELITE_K)
            elite_idx = sorted(range(POP_SIZE), key=lambda i: fitnesses[i])[:elite_k]
            elites = [duplicate_individual(population[i]) for i in elite_idx]

            # Elite fitness (batched)
            ELITE_CHUNK = None
            elite_fits = fitness_population(elites, target, tile=32, chunk=ELITE_CHUNK, weight_mask=imp_mask)

            # Next generation
            population = elites + offspring[:POP_SIZE-ELITE_K]
            fitnesses  = elite_fits + off_fits[:POP_SIZE-ELITE_K]

            gbest_idx = min(range(POP_SIZE), key=lambda i: fitnesses[i])
            if fitnesses[gbest_idx] + 1e-10 < best_fit:
                best_fit = fitnesses[gbest_idx]
                best_ind = duplicate_individual(population[gbest_idx])
                no_improve = 0
            else:
                no_improve += 1

            if SAVE_VIDEO and (gen % max(1, FRAME_EVERY) == 0):
                save_frame_png(gen, best_ind, pad, prefix="ga")

            f = _anneal_factor(gen, GENERATIONS, SCHEDULE)
            pbar.set_postfix(best_mse=f"{best_fit:.6f}", stale=no_improve, sigma_fac=f"{f:.3f}")

    except KeyboardInterrupt:
        try: pbar.close()
        except Exception: pass
        print("\n[Interrupted] Returning current best individual…", flush=True)
    finally:
        try: pbar.close()
        except Exception: pass

    return best_ind.cpu(), best_fit

# ============================================================
# Example usage
# ============================================================
if __name__ == "__main__":
    img_path = os.path.join(INPUT_DIR, "reference.jpg")
    pil_img = Image.open(img_path).convert("RGB")
    np_img = np.array(pil_img, dtype=np.float32) / 255.0
    target_img = torch.from_numpy(np_img)

    H_out, W_out = target_img.shape[0], target_img.shape[1]
    H, W = choose_work_size(H_out, W_out, max_side=WORK_MAX_SIDE)

    best_ind, best_fit = genetic_approx(target_img)
    print("Best MSE (working res):", best_fit)

    if best_ind is not None:
        sH = H_out / float(H); sW = W_out / float(W)
        best_ind_full = scale_genome_pixels_anisotropic(best_ind.to(DEV), sH=sH, sW=sW)
        best_ind_full_render = genome_to_renderer(best_ind_full)
        if best_ind_full_render.shape[1] > 9:
            best_ind_full_render = best_ind_full_render[:, :9]
        if best_ind_full_render.shape[1] < 9:
            pad = torch.full((best_ind_full_render.shape[0],1), 255.0, device=best_ind_full_render.device, dtype=best_ind_full_render.dtype)
            best_ind_full_render = torch.cat([best_ind_full_render, pad], dim=1)

        final = render_splats_rgb_triton(best_ind_full_render.unsqueeze(0), H_out, W_out, k_sigma=K_SIGMA, device=DEV, tile=32)[0]
        img8 = (final.clamp(0,1).detach().cpu().numpy() * 255).astype('uint8')
        out_path = os.path.join(OUTPUT_DIR, "ga_splats.png")
        Image.fromarray(img8).save(out_path)
        print("Saved full-resolution result as ga_splats.png")

        if SAVE_VIDEO:
            print(f"Saved frames (every {FRAME_EVERY} gens) to {VIDEO_DIR}")
        else:
            print("Frame saving disabled (SAVE_VIDEO=False).")
