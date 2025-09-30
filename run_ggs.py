from __future__ import annotations

import os, math
import numpy as np
import torch, random
import torch.nn.functional as F
from typing import Tuple, List
from PIL import Image
from tqdm.auto import tqdm

from modules.render import render_splats_rgb_triton, _DEV as DEV
from modules.resize import choose_work_size, scale_genome_pixels_anisotropic
from modules.encode import genome_to_renderer, genome_to_renderer_batched
from modules.mask import compute_importance_mask


INPUT_DIR    = "imgs"
OUTPUT_DIR   = "output"
REF_IMG     = "reference.jpg"
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

if SEED is not None:
    random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

MIN_SCALE_SPLATS = 0.8
MAX_SCALE_SPLATS = 0.15

# MUT_SIGMA_MAX = {"xy":0.1, "alog":0.5, "blog":0.5, "theta":0.3,
#                  "rgb":20.0, "alpha":20.0}
# MUT_SIGMA_MIN = {"xy":0.005,"alog":0.01, "blog":0.01, "theta":0.01,
#                  "rgb":1.0, "alpha":1.0}

MUT_SIGMA_MAX = {"xy":0.1, "alog":0.5, "blog":0.5, "theta":0.3,
                 "rgb":25.0, "alpha":25.0}
MUT_SIGMA_MIN = {"xy":0.01,"alog":0.05, "blog":0.05, "theta":0.025,
                 "rgb":5.0, "alpha":5.0}

SCHEDULE = "cosine"  # "linear", "cosine", "exp"

MASK_STRENGTH = 0.7  # 1.0 = full, 0.0 = plain MSE
BOOST_ONLY    = False  # If True, use mask to only boost important areas over plain MSE


@torch.no_grad()
def render_axes_angle_to_img(ind_axes_angle: torch.Tensor, Hsnap:int, Wsnap:int) -> np.ndarray:
    G = ind_axes_angle.unsqueeze(0) if ind_axes_angle.ndim == 2 else ind_axes_angle  # [1,N,C]
    G9 = genome_to_renderer_batched(G)  # [1,N,9]
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
    dummy = torch.tensor([[[0.5,0.5, math.log(2.0), math.log(2.0), 0.0, 128.0,128.0,128.0, 255.0]]],
                         device=device, dtype=torch.float32)  # [1,1,9]
    _ = render_splats_rgb_triton(dummy, min(8,H), min(8,W), k_sigma=K_SIGMA, device=device, tile=32)
    _ = render_splats_rgb_triton(dummy, min(8,H), min(8,W), k_sigma=K_SIGMA, device=device, tile=32)
    if torch.cuda.is_available():
        torch.cuda.synchronize()


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
    B, N = batch_size, n_splats
    max_side = float(max(H, W))

    # x, y in [0,1]
    xy = torch.empty(B, N, 2, device=device, dtype=dtype).uniform_(0.0, 1.0)

    # Scales (sample in linear-sigma, then log)
    s_lo = float(MIN_SCALE_SPLATS)
    s_hi = float(MAX_SCALE_SPLATS * max_side)
    a = torch.empty(B, N, 1, device=device, dtype=dtype).uniform_(s_lo, s_hi).log()
    b = torch.empty(B, N, 1, device=device, dtype=dtype).uniform_(s_lo, s_hi).log()

    # Theta
    theta = torch.empty(B, N, 1, device=device, dtype=dtype).uniform_(-math.pi, math.pi)

    # Colors + alpha
    rgb   = torch.empty(B, N, 3, device=device, dtype=dtype).uniform_(0.0, 256.0)
    alpha = torch.empty(B, N, 1, device=device, dtype=dtype).uniform_(180.0, 256.0)

    G = torch.cat([xy, a, b, theta, rgb, alpha], dim=-1)  # [B,N,9]
    G[..., 0:2].clamp_(0.0, 1.0)
    G[..., 5:9].clamp_(0.0, 255.0)
    return G


def new_individual(n_splats=N_SPLATS, device=DEV) -> torch.Tensor:
    return new_population(1, n_splats, H, W, device=device)[0]


def duplicate_individual(ind:torch.Tensor) -> torch.Tensor:
    return ind.clone()


@torch.no_grad()
def fitness_many(pop_batch, target, tile: int = 32,
                 weight_mask: torch.Tensor | None = None,
                 boost_only: bool = False,
                 boost_beta: float = 1.0):
    G_axes = torch.stack(pop_batch, dim=0)  # [B,N,C]
    G9 = genome_to_renderer_batched(G_axes)  # [B,N,9]
    imgs = render_splats_rgb_triton(G9, H, W, k_sigma=K_SIGMA, device=DEV, tile=tile)  # [B,H,W,3]
    dif2 = (imgs - target.unsqueeze(0)) ** 2  # [B,H,W,3]

    if weight_mask is None:
        return dif2.mean(dim=(1,2,3))

    w = weight_mask.unsqueeze(0).unsqueeze(-1)  # [1,H,W,1]

    if boost_only:
        w_boost = 1.0 + boost_beta * w.clamp(0, 1)
        num = (dif2 * w_boost).mean(dim=(1,2,3))
        den = w_boost.mean(dim=(1,2,3)) + 1e-12
        return num / den
    else:
        num = (dif2 * w).sum(dim=(1,2,3))
        den = (w.sum(dim=(1,2,3)) + 1e-12)
        return num / den


@torch.no_grad()
def fitness_population(population: List[torch.Tensor], target: torch.Tensor,
                       tile: int = 32, chunk: int | None = None,
                       weight_mask: torch.Tensor | None = None) -> List[float]:
    if chunk is None or chunk >= len(population):
        return fitness_many(population, target, tile=tile, 
                            weight_mask=weight_mask, boost_only=BOOST_ONLY).detach().cpu().tolist()
    out: List[float] = []
    for i in range(0, len(population), chunk):
        out.extend(fitness_many(population[i:i+chunk], target, tile=tile, 
                                weight_mask=weight_mask, boost_only=BOOST_ONLY).detach().cpu().tolist())
    return out


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

        # Masks for which genes to mutate
        m_xy = (torch.rand((N, 2), device=ind.device) < MUTPB)
        m_ab = (torch.rand((N, 2), device=ind.device) < MUTPB)
        m_t  = (torch.rand((N, 1), device=ind.device) < MUTPB)

        # Color + alpha mutation, ensure at least one of RGBA mutates
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

        # Numeric mutations
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

        # Swap two splats
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


# Main genetic algorithm loop
@torch.no_grad()
def genetic_approx(target_img_uint8: torch.Tensor) -> Tuple[torch.Tensor, float]:
    t = target_img_uint8.to(torch.float32)
    if t.max() > 1.5: t = t / 255.0
    if t.shape[0] != H or t.shape[1] != W:
        tBCHW = t.permute(2,0,1).unsqueeze(0)
        t = F.interpolate(tBCHW, size=(H,W), mode='bilinear', align_corners=False)[0].permute(1,2,0)
    t = t.contiguous()

    imp_mask = compute_importance_mask(
        t, H, W,
        edge_scales=(1,2,4),
        w_edge=0.7, w_var=0.3,
        gamma=0.7, floor=0.15,
        smooth=3,
        strength=MASK_STRENGTH
    ).to(DEV) 

    target = t.to(DEV)
    prewarm_renderer(H, W, device=DEV)

    pop_tensor = new_population(POP_SIZE, N_SPLATS, H, W, device=DEV)  # [P,N,9]
    population = [pop_tensor[i] for i in range(POP_SIZE)]

    fitnesses  = fitness_population(population, target, tile=32, chunk=None, weight_mask=imp_mask)

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

            # Offspring fitness
            OFFSPRING_CHUNK = None
            off_fits = fitness_population(offspring, target, tile=32, chunk=OFFSPRING_CHUNK, weight_mask=imp_mask)

            # Elites
            elite_k = max(1, ELITE_K)
            elite_idx = sorted(range(POP_SIZE), key=lambda i: fitnesses[i])[:elite_k]
            elites = [duplicate_individual(population[i]) for i in elite_idx]

            # Elite fitness
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


if __name__ == "__main__":
    img_path = os.path.join(INPUT_DIR, REF_IMG)
    pil_img = Image.open(img_path).convert("RGB")
    np_img = np.array(pil_img, dtype=np.float32) / 255.0
    target_img = torch.from_numpy(np_img)

    H_out, W_out = target_img.shape[0], target_img.shape[1]
    H, W = choose_work_size(H_out, W_out, max_side=WORK_MAX_SIDE)

    best_ind, best_fit = genetic_approx(target_img)
    print("Best MSE:", best_fit)

    if best_ind is not None:
        sH = H_out / float(H); sW = W_out / float(W)
        best_ind_full = scale_genome_pixels_anisotropic(best_ind.to(DEV), sH=sH, sW=sW)
        best_ind_full_render = genome_to_renderer(best_ind_full)

        final = render_splats_rgb_triton(best_ind_full_render.unsqueeze(0), H_out, W_out, 
                                         k_sigma=K_SIGMA, device=DEV, tile=32)[0]
        img8 = (final.clamp(0,1).detach().cpu().numpy() * 255).astype('uint8')
        out_path = os.path.join(OUTPUT_DIR, "ga_splats.png")

        Image.fromarray(img8).save(out_path)
        print("Saved full-resolution result as ga_splats.png")

        if SAVE_VIDEO:
            print(f"Saved frames to {VIDEO_DIR}")
