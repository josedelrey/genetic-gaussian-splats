from __future__ import annotations

import os, math
import numpy as np
import torch, random
from typing import Tuple, List
from PIL import Image
from tqdm.auto import tqdm

# ⬇️ Use your **batched** Triton renderer (the one you pasted)
from modules.render_triton import render_splats_rgb_triton, _DEV as DEV
from modules.resize import choose_work_size, scale_genome_pixels_anisotropic
from modules.encode import genome_to_renderer  # single-sample; we wrap it batched

# ============================================================
# Config
# ============================================================
INPUT_DIR    = "imgs"
OUTPUT_DIR   = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Video controls ---
SAVE_VIDEO   = True
FRAME_EVERY  = 100

VIDEO_DIR    = os.path.join(OUTPUT_DIR, "video_frames")
if SAVE_VIDEO:
    os.makedirs(VIDEO_DIR, exist_ok=True)

H, W          = 128, 128
WORK_MAX_SIDE = 512
N_SPLATS      = 512
POP_SIZE      = 64
GENERATIONS   = 100000
TOUR_K        = 2
ELITE_K       = max(1, POP_SIZE // 10)
CXPB          = 0.7

# Mutation probabilities
MUTPB_NON_ELITE  = 0.2
MUTPB_ELITE      = 0.1
ELITE_MUT_FRAC   = 0.9
PROTECT_BEST_ELITE = True
ELITE_SIGMA_MULT = 0.7

K_SIGMA       = 3.0
SEED          = 42

# Genome clamping
MIN_SCALE_SPLATS = 1.0
MAX_SCALE_SPLATS = 0.2

# Annealed mutation SIGMA ranges
MUT_SIGMA_MAX = {"xy":0.05,"alog":0.25,"blog":0.25,"theta":0.35,"rgb":8.0,"alpha":8.0}
MUT_SIGMA_MIN = {"xy":0.005,"alog":0.05,"blog":0.05,"theta":0.05,"rgb":2.0,"alpha":2.0}

# Debug checks that would sync host/device if enabled
DEBUG_CHECKS = False

# Repro
if SEED is not None:
    random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

# ============================================================
# Annealing utils
# ============================================================
def _anneal_factor(gen, total, kind="cosine"):
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
    return max(0.2, raw)

def build_mut_sigma(gen:int, total_gens:int, kind:str="cosine"):
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
    a = torch.empty(B, N, 1, device=device, dtype=dtype).uniform_(s_lo, s_hi).log_()
    b = torch.empty(B, N, 1, device=device, dtype=dtype).uniform_(s_lo, s_hi).log_()

    # theta
    theta = torch.empty(B, N, 1, device=device, dtype=dtype).uniform_(-math.pi, math.pi)

    # colors + alpha (uint8 range but kept float here)
    rgb   = torch.empty(B, N, 3, device=device, dtype=dtype).uniform_(0.0, 256.0)
    alpha = torch.empty(B, N, 1, device=device, dtype=dtype).uniform_(64.0, 256.0)

    G = torch.cat([xy, a, b, theta, rgb, alpha], dim=-1)  # [B,N,9]
    # Clamp once (angle wrap optional here; theta is already in-range)
    G[..., 0:2].clamp_(0.0, 1.0)
    G[..., 5:9].clamp_(0.0, 255.0)
    return G

def new_individual(n_splats=N_SPLATS, device=DEV) -> torch.Tensor:
    # Keep signature used elsewhere; just slice B=1 result.
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

    if DEBUG_CHECKS:
        ok = torch.isfinite(R).all()
        if not bool(ok.item()):
            raise RuntimeError("Non-finite values in renderer genome.")
    return R.reshape(B, N, 9)

# ============================================================
# Rendering helpers (use the SAME batched renderer)
# ============================================================
@torch.no_grad()
def render_axes_angle_to_img(ind_axes_angle: torch.Tensor, Hsnap:int, Wsnap:int) -> np.ndarray:
    """
    Renders a single individual using the batched renderer (B=1) for consistency.
    """
    G = ind_axes_angle.unsqueeze(0) if ind_axes_angle.ndim == 2 else ind_axes_angle  # [1,N,C]
    G9 = genome_to_renderer_batched(G)                                               # [1,N,9]
    # No assert here; avoid host sync on first call
    img = render_splats_rgb_triton(G9, Hsnap, Wsnap, k_sigma=K_SIGMA, device=DEV, tile=64)   # [1,H,W,3]
    img = img[0]                                                                     # [H,W,3]
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
    # Warm two common tile sizes so the first "real" render doesn't compile
    _ = render_splats_rgb_triton(dummy, min(8,H), min(8,W), k_sigma=K_SIGMA, device=device, tile=32)
    _ = render_splats_rgb_triton(dummy, min(8,H), min(8,W), k_sigma=K_SIGMA, device=device, tile=64)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

# ============================================================
# Fitness (batched everywhere)
# ============================================================
@torch.no_grad()
def fitness_many(pop_batch: List[torch.Tensor], target: torch.Tensor, tile: int = 32) -> torch.Tensor:
    """
    pop_batch: list of individuals in axes+theta space, each [N,8/9] on CUDA.
    target:    [H,W,3] on CUDA in [0,1]
    returns:   [B] CUDA tensor of MSE per individual
    """
    G_axes = torch.stack(pop_batch, dim=0)                 # [B,N,C]
    G9 = genome_to_renderer_batched(G_axes)                # [B,N,9]
    # No assert (avoids sync); rely on DEBUG_CHECKS in converter if needed
    imgs = render_splats_rgb_triton(G9, H, W, k_sigma=K_SIGMA, device=DEV, tile=tile)  # [B,H,W,3]
    dif = imgs - target.unsqueeze(0)
    return (dif * dif).mean(dim=(1,2,3))                   # [B]

@torch.no_grad()
def fitness_population(population: List[torch.Tensor], target: torch.Tensor, tile: int = 32, chunk: int | None = None) -> List[float]:
    """
    Evaluate the whole population in CUDA batches. If chunk is None, do it in one go.
    Returns a Python list of floats.
    """
    if chunk is None or chunk >= len(population):
        return fitness_many(population, target, tile=tile).detach().cpu().tolist()
    out: List[float] = []
    for i in range(0, len(population), chunk):
        out.extend(fitness_many(population[i:i+chunk], target, tile=tile).detach().cpu().tolist())
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

def crossover_swap_splats(a:torch.Tensor, b:torch.Tensor):
    if a.shape[0] != b.shape[0]:
        raise ValueError("Different N_SPLATS")
    n = a.shape[0]
    cx = random.randrange(1, n)
    child1 = torch.vstack([a[:cx], b[cx:]])
    child2 = torch.vstack([b[:cx], a[cx:]])
    return child1, child2

def mutate_individual(ind, is_elite, gen, total_gens, schedule="cosine"):
    MUTPB = MUTPB_ELITE if is_elite else MUTPB_NON_ELITE
    SIG = build_mut_sigma(gen, total_gens, schedule)
    if is_elite:
        SIG = {k: v * ELITE_SIGMA_MULT for k, v in SIG.items()}

    with torch.no_grad():
        m_xy = torch.rand((ind.shape[0],2), device=ind.device) < MUTPB
        ind[:,0:2] += torch.randn_like(ind[:,0:2]) * SIG["xy"] * m_xy.float()

        m_ab = torch.rand((ind.shape[0],2), device=ind.device) < MUTPB
        ind[:,2:4] += torch.randn_like(ind[:,2:4]) * torch.tensor([SIG["alog"], SIG["blog"]], device=ind.device) * m_ab.float()

        m_t = (torch.rand((ind.shape[0],1), device=ind.device) < MUTPB).float()
        ind[:,4:5] += torch.randn_like(ind[:,4:5]) * SIG["theta"] * m_t
        ind[:,4] = wrap_angle(ind[:,4])

        m_rgba = torch.rand((ind.shape[0],4), device=ind.device) < MUTPB
        sig_rgba = torch.tensor([SIG["rgb"],SIG["rgb"],SIG["rgb"],SIG["alpha"]], device=ind.device)
        ind[:,5:9] += torch.randn_like(ind[:,5:9]) * sig_rgba * m_rgba.float()

    return clamp_genome(ind)

# ============================================================
# Main GA loop (with tqdm + frame export)
# ============================================================
@torch.no_grad()
def genetic_approx(target_img_uint8: torch.Tensor) -> Tuple[torch.Tensor, float]:
    t = target_img_uint8.to(torch.float32)
    if t.max() > 1.5: t = t / 255.0
    if t.shape[0] != H or t.shape[1] != W:
        tBCHW = t.permute(2,0,1).unsqueeze(0)
        t = torch.nn.functional.interpolate(tBCHW, size=(H,W), mode='bilinear', align_corners=False)[0].permute(1,2,0)
    target = t.contiguous().to(DEV)

    # ---- Pre-warm CUDA + Triton JIT (fast tiny render) ----
    prewarm_renderer(H, W, device=DEV)

    # ---- Init population on CUDA (vectorized, single call) ----
    pop_tensor = new_population(POP_SIZE, N_SPLATS, H, W, device=DEV)  # [P,N,9]
    population = [pop_tensor[i] for i in range(POP_SIZE)]

    # ---- Fitness (batched) ----
    INIT_CHUNK = None  # set to e.g. 32 if VRAM limited
    fitnesses  = fitness_population(population, target, tile=64, chunk=INIT_CHUNK)

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

            offspring = []
            for i in range(0, POP_SIZE, 2):
                a = parents[i]
                b = parents[(i+1) % POP_SIZE]
                if random.random() < CXPB:
                    c1, c2 = crossover_swap_splats(a, b)
                else:
                    c1, c2 = duplicate_individual(a), duplicate_individual(b)
                offspring.append(mutate_individual(c1, is_elite=False, gen=gen, total_gens=GENERATIONS))
                if len(offspring) < POP_SIZE:
                    offspring.append(mutate_individual(c2, is_elite=False, gen=gen, total_gens=GENERATIONS))

            # Offspring fitness (batched)
            OFFSPRING_CHUNK = None
            off_fits = fitness_population(offspring, target, tile=64, chunk=OFFSPRING_CHUNK)

            # Elites
            elite_idx = sorted(range(POP_SIZE), key=lambda i: fitnesses[i])[:ELITE_K]
            elites = [duplicate_individual(population[i]) for i in elite_idx]

            for _, elite in enumerate(elites):
                if PROTECT_BEST_ELITE and torch.allclose(elite, best_ind):
                    continue
                if random.random() < ELITE_MUT_FRAC:
                    mutate_individual(elite, is_elite=True, gen=gen, total_gens=GENERATIONS)

            # Elite fitness (batched)
            ELITE_CHUNK = None
            elite_fits = fitness_population(elites, target, tile=64, chunk=ELITE_CHUNK)

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

            f = _anneal_factor(gen, GENERATIONS, "cosine")
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

        # Use the same batched renderer with B=1 for the final image
        final = render_splats_rgb_triton(best_ind_full_render.unsqueeze(0), H_out, W_out, k_sigma=K_SIGMA, device=DEV, tile=64)[0]
        img8 = (final.clamp(0,1).detach().cpu().numpy() * 255).astype('uint8')
        out_path = os.path.join(OUTPUT_DIR, "ga_splats.png")
        Image.fromarray(img8).save(out_path)
        print("Saved full-resolution result as ga_splats.png")

        if SAVE_VIDEO:
            print(f"Saved frames (every {FRAME_EVERY} gens) to {VIDEO_DIR}")
        else:
            print("Frame saving disabled (SAVE_VIDEO=False).")
