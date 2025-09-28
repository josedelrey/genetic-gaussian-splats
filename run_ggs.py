from __future__ import annotations

import os, math
import numpy as np
import torch, random
from typing import Tuple
from PIL import Image
from tqdm.auto import tqdm

from modules.render import render_splats_rgb, _DEV as DEV
from modules.resize import choose_work_size, scale_genome_pixels_anisotropic
from modules.encode import genome_to_renderer

# ============================================================
# Config
# ============================================================
INPUT_DIR    = "imgs"
OUTPUT_DIR   = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

H, W          = 128, 128          # initial placeholder (overwritten by choose_work_size)
WORK_MAX_SIDE = 64                # working resolution max side for GA
N_SPLATS      = 256
POP_SIZE      = 32
GENERATIONS   = 1000
TOUR_K        = 2
ELITE_K       = max(1, POP_SIZE // 10)   # keep at least 1 elite
CXPB          = 0.7

# Mutation probabilities
MUTPB_NON_ELITE  = 0.2          # per-gene mutation probability for non-elites
MUTPB_ELITE      = 0.3          # per-gene mutation probability for elites
ELITE_MUT_FRAC   = 0.9          # 0..1; e.g., 0.9 means mutate 90% of elites
PROTECT_BEST_ELITE = True       # do not mutate the overall best individual
ELITE_SIGMA_MULT = 0.7          # elites get smaller mutation steps (polish)

K_SIGMA       = 3.0
SEED          = 42              # set to None for non-deterministic runs

# Genome clamping (sizes are in pixels before logging)
MIN_SCALE_SPLATS = 1.0            # min radius in pixels
MAX_SCALE_SPLATS = 0.10           # max radius relative to max(H,W)

# -------- Annealed mutation SIGMA ranges (MAX -> MIN over time) --------
MUT_SIGMA_MAX = {
    "xy":    0.06,   # normalized coords (start broader)
    "alog":  0.40,   # logs in pixel units
    "blog":  0.40,
    "theta": 0.70,   # radians
    "rgb":   30.0,   # 0..255
}
MUT_SIGMA_MIN = {
    "xy":    0.01,
    "alog":  0.06,
    "blog":  0.06,
    "theta": 0.03,
    "rgb":   4.0,
}

# Reproducibility
if SEED is not None:
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

# ============================================================
# Annealing utilities
# ============================================================
def _anneal_factor(gen:int, total_gens:int, kind:str="cosine") -> float:
    """
    Factor in [0,1], where 1.0 at gen=0 and ~0.0 at gen=total_gens.
    Schedules: 'linear' | 'cosine' | 'exp'
    """
    g = max(0, min(gen, total_gens))
    p = g / max(1, total_gens)

    if kind == "linear":
        f = 1.0 - p
    elif kind == "cosine":
        # Smooth start & end, keeps exploration a bit longer
        f = 0.5 * (1.0 + math.cos(math.pi * p))
    elif kind == "exp":
        # Pick a target floor at the end (e.g., 0.05)
        target = 0.05
        decay = target ** (1.0 / max(1, total_gens))
        f = decay ** g
    else:
        f = 1.0 - p
    return max(0.0, min(1.0, f))

def build_mut_sigma(gen:int, total_gens:int, kind:str="cosine"):
    """Sigma per key: MIN + f*(MAX - MIN), where f∈[0,1]."""
    f = _anneal_factor(gen, total_gens, kind)
    return {
        k: MUT_SIGMA_MIN[k] + f * (MUT_SIGMA_MAX[k] - MUT_SIGMA_MIN[k])
        for k in MUT_SIGMA_MAX.keys()
    }

# ============================================================
# Genome helpers
# Each splat row: [x, y, a_log, b_log, theta, r, g, b]  -> total 8 entries
# ============================================================
def random_splat(H:int, W:int, device=DEV) -> torch.Tensor:
    # sizes (in pixels) then log
    a_log = torch.log(torch.tensor(float(random.uniform(3, 20)), device=device))
    b_log = torch.log(torch.tensor(float(random.uniform(3, 20)), device=device))
    theta = torch.tensor(float(random.uniform(-np.pi, np.pi)), device=device)
    rgb   = torch.tensor([random.uniform(0,255) for _ in range(3)], device=device)
    return torch.tensor([
        random.random(),            # x in [0,1]
        random.random(),            # y in [0,1]
        a_log.item(),
        b_log.item(),
        theta.item(),
        rgb[0].item(), rgb[1].item(), rgb[2].item(),
    ], device=device, dtype=torch.float32)

@torch.no_grad()
def wrap_angle(theta: torch.Tensor) -> torch.Tensor:
    # wrap to (-pi, pi]
    return (theta + np.pi) % (2*np.pi) - np.pi

def clamp_genome(ind:torch.Tensor) -> torch.Tensor:
    """
    Clamp/wrap genome (with theta at column 4). No alpha anywhere.
    """
    # x,y
    ind[:,0:2] = ind[:,0:2].clamp(0.0, 1.0)

    # a_log, b_log bounds
    max_side = float(max(H, W))
    min_scale_log = torch.log(torch.tensor(MIN_SCALE_SPLATS, device=ind.device))
    max_scale_log = torch.log(torch.tensor(MAX_SCALE_SPLATS * max_side, device=ind.device))
    ind[:,2] = ind[:,2].clamp(min_scale_log, max_scale_log)
    ind[:,3] = ind[:,3].clamp(min_scale_log, max_scale_log)

    # theta: wrap, do NOT clamp
    ind[:,4] = wrap_angle(ind[:,4])

    # rgb
    ind[:,5:8] = ind[:,5:8].clamp(0.0, 255.0)

    return ind

def new_individual(n_splats=N_SPLATS, device=DEV) -> torch.Tensor:
    rows = [random_splat(H, W, device=device) for _ in range(n_splats)]
    return clamp_genome(torch.stack(rows, dim=0))  # [N,8]

def duplicate_individual(ind:torch.Tensor) -> torch.Tensor:
    return ind.clone()

# ============================================================
# Fitness: MSE to target (target expected in [0,1], HxW resized)
# ============================================================
@torch.no_grad()
def fitness_mse(ind_axes_angle:torch.Tensor, target:torch.Tensor) -> float:
    # Convert to renderer format (with Cholesky params, no alpha)
    ind_render = genome_to_renderer(ind_axes_angle).to(DEV)
    pred = render_splats_rgb(ind_render, H, W, k_sigma=K_SIGMA, device=DEV)  # [H,W,3], in [0,1]
    pred = pred.to(target.device)
    return torch.mean((pred - target)**2).item()

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
    """One-point crossover at the *row* (splat) level."""
    if a.shape[0] != b.shape[0]:
        raise ValueError("Different N_SPLATS")
    n = a.shape[0]
    cx = random.randrange(1, n)   # split after cx-1
    child1 = torch.vstack([a[:cx], b[cx:]])
    child2 = torch.vstack([b[:cx], a[cx:]])
    return child1, child2

def mutate_individual(ind:torch.Tensor,
                      is_elite:bool,
                      gen:int,
                      total_gens:int,
                      schedule:str="cosine") -> torch.Tensor:
    """
    Mutation with annealed sigmas. Elites get smaller steps via ELITE_SIGMA_MULT.
    """
    MUTPB = MUTPB_ELITE if is_elite else MUTPB_NON_ELITE
    SIG = build_mut_sigma(gen, total_gens, schedule)
    if is_elite:
        SIG = {k: v * ELITE_SIGMA_MULT for k, v in SIG.items()}

    with torch.no_grad():
        # x, y
        mask = torch.rand_like(ind[:,0]) < MUTPB
        ind[mask,0] += torch.randn((mask.sum(),), device=ind.device) * SIG["xy"]
        mask = torch.rand_like(ind[:,1]) < MUTPB
        ind[mask,1] += torch.randn((mask.sum(),), device=ind.device) * SIG["xy"]

        # a_log, b_log
        mask = torch.rand_like(ind[:,2]) < MUTPB
        ind[mask,2] += torch.randn((mask.sum(),), device=ind.device) * SIG["alog"]
        mask = torch.rand_like(ind[:,3]) < MUTPB
        ind[mask,3] += torch.randn((mask.sum(),), device=ind.device) * SIG["blog"]

        # theta (wrap)
        mask = torch.rand_like(ind[:,4]) < MUTPB
        ind[mask,4] += torch.randn((mask.sum(),), device=ind.device) * SIG["theta"]
        ind[:,4] = wrap_angle(ind[:,4])

        # rgb
        for c in (5,6,7):
            mask = torch.rand_like(ind[:,c]) < MUTPB
            ind[mask,c] += torch.randn((mask.sum(),), device=ind.device) * SIG["rgb"]

    return clamp_genome(ind)

# ============================================================
# Main GA loop (with tqdm)
# ============================================================
@torch.no_grad()
def genetic_approx(target_img_uint8: torch.Tensor) -> Tuple[torch.Tensor, float]:
    """
    target_img_uint8: [H,W,3] or larger; values 0..255 uint8/float.
    Returns (best_individual_axes_angle, best_fitness) at the *working* resolution (H,W).
    """
    # Prepare target at working res in [0,1]
    t = target_img_uint8.to(torch.float32)
    if t.max() > 1.5: t = t / 255.0
    if t.shape[0] != H or t.shape[1] != W:
        # simple resize using bilinear on BCHW
        tBCHW = t.permute(2,0,1).unsqueeze(0)
        t = torch.nn.functional.interpolate(
            tBCHW, size=(H,W), mode='bilinear', align_corners=False
        )[0].permute(1,2,0)
    target = t.contiguous().to(DEV)

    # --- Initialize population
    population = [new_individual(N_SPLATS, device=DEV) for _ in range(POP_SIZE)]
    fitnesses  = [fitness_mse(ind, target) for ind in population]

    best_idx = min(range(POP_SIZE), key=lambda i: fitnesses[i])
    best_ind, best_fit = duplicate_individual(population[best_idx]), fitnesses[best_idx]
    no_improve = 0

    pbar = tqdm(range(GENERATIONS), desc="GA generations", leave=True)
    try:
        for gen in pbar:
            # -- selection
            parents = []
            while len(parents) < POP_SIZE:
                parents.append(tournament_selection(population, fitnesses, k=TOUR_K))

            # -- crossover + mutation
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

            # -- evaluate offspring
            off_fits = [fitness_mse(ind, target) for ind in offspring]

            # -- elites
            elite_idx = sorted(range(POP_SIZE), key=lambda i: fitnesses[i])[:ELITE_K]
            elites = [duplicate_individual(population[i]) for i in elite_idx]

            for _, elite in enumerate(elites):
                # protect the global best if requested
                if PROTECT_BEST_ELITE and torch.allclose(elite, best_ind):
                    continue
                if random.random() < ELITE_MUT_FRAC:
                    mutate_individual(elite, is_elite=True, gen=gen, total_gens=GENERATIONS)

            elite_fits = [fitnesses[i] for i in elite_idx]

            # -- replacement
            population = elites + offspring[:POP_SIZE-ELITE_K]
            fitnesses  = elite_fits + off_fits[:POP_SIZE-ELITE_K]

            # -- track best + convergence
            gbest_idx = min(range(POP_SIZE), key=lambda i: fitnesses[i])
            if fitnesses[gbest_idx] + 1e-10 < best_fit:
                best_fit = fitnesses[gbest_idx]
                best_ind = duplicate_individual(population[gbest_idx])
                no_improve = 0
            else:
                no_improve += 1

            # (optional) show current SIGMA factor in tqdm
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
    img_path = os.path.join(INPUT_DIR, "dog.jpg")
    pil_img = Image.open(img_path).convert("RGB")
    np_img = np.array(pil_img, dtype=np.float32) / 255.0
    target_img = torch.from_numpy(np_img)

    H_out, W_out = target_img.shape[0], target_img.shape[1]
    H, W = choose_work_size(H_out, W_out, max_side=WORK_MAX_SIDE)

    best_ind, best_fit = genetic_approx(target_img)
    print("Best MSE (working res):", best_fit)

    if best_ind is not None:
        # scale genome (axes+angle) to full res
        sH = H_out / float(H)
        sW = W_out / float(W)
        best_ind_full = scale_genome_pixels_anisotropic(best_ind.to(DEV), sH=sH, sW=sW)

        # convert to renderer format (Cholesky); strip any alpha column if present
        best_ind_full_render = genome_to_renderer(best_ind_full)
        if best_ind_full_render.shape[1] >= 9:
            best_ind_full_render = best_ind_full_render[:, :8]

        # final render at full resolution
        final_img = render_splats_rgb(best_ind_full_render, H_out, W_out, k_sigma=K_SIGMA, device=DEV)
        img8 = (final_img.clamp(0,1).cpu().numpy() * 255).astype('uint8')
        out_path = os.path.join(OUTPUT_DIR, "ga_splats.png")
        Image.fromarray(img8).save(out_path)
        print("Saved full-resolution result as ga_splats.png")
