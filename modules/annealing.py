from __future__ import annotations
import torch
import math
import random
import torch.nn.functional as F
from tqdm.auto import tqdm
from typing import Tuple

from modules.population import new_individual, duplicate_individual
from modules.fitness import fitness_population
from modules.genetic import mutate_individual
from modules.mask import compute_importance_mask
from modules.utils import (
    prewarm_renderer, save_frame_png, _anneal_factor,
    save_loss_curve_png, save_curves_csv
)


@torch.no_grad()
def _ensure_hw(target_img_uint8: torch.Tensor, H:int, W:int) -> torch.Tensor:
    t = target_img_uint8.to(torch.float32)
    if t.max() > 1.5: t = t / 255.0
    if t.shape[0] != H or t.shape[1] != W:
        tBCHW = t.permute(2,0,1).unsqueeze(0)
        t = F.interpolate(tBCHW, size=(H,W), mode="bilinear", align_corners=False)[0].permute(1,2,0)
    return t.contiguous()


def _temp_schedule(kind:str, T0:float, i:int, total:int) -> float:
    """Temperature schedule T(i). i starts at 0."""
    p = i / max(1, total)
    if kind == "exp":           # geometric
        # decays to ~1% by the end
        r = 0.01 ** (1.0 / max(1,total))
        return T0 * (r ** i)
    elif kind == "linear":
        return max(1e-12, T0 * (1.0 - p))
    elif kind == "cosine":
        return max(1e-12, T0 * 0.5 * (1.0 + math.cos(math.pi * p)))
    elif kind == "log":
        return max(1e-12, T0 / (1.0 + math.log(1.0 + 9.0 * i)))
    elif kind == "cauchy":
        return max(1e-12, T0 / (1.0 + i))
    else:  # default: exp
        r = 0.01 ** (1.0 / max(1,total))
        return T0 * (r ** i)


@torch.no_grad()
def simulated_annealing(
    target_img_uint8: torch.Tensor,
    H:int, W:int, device,

    # genome / neighborhood
    n_splats:int,
    mutpb:float,
    mut_sigma_max:dict,
    mut_sigma_min:dict,
    sigma_schedule:str,              # reuse your GA sigma schedule ("linear","cosine","exp")
    min_scale_splats:float,
    max_scale_splats:float,

    # renderer / fitness
    k_sigma:float,
    mask_strength:float,
    boost_only:bool,

    # SA loop
    iterations:int,
    temp0:float,
    temp_schedule:str,               # "exp","linear","cosine","log","cauchy"
    tries_per_iter:int = 1,          # how many neighbors per T step

    # saving / logs
    save_video:bool = False,
    frame_every:int = 10_000,
    video_dir:str = "",
    prefix:str = "sa",
    loss_png_path:str = "",
    loss_csv_path:str = "",
    loss_log_y:bool = False,
) -> Tuple[torch.Tensor, float]:
    """
    Simulated Annealing main loop. Minimizes MSE energy.
    Returns (best_individual_axes_angle, best_mse).
    """

    # --- prep image + importance mask ---
    t = _ensure_hw(target_img_uint8, H, W)
    target = t.to(device)
    imp_mask = compute_importance_mask(
        t, H, W,
        edge_scales=(1,2,4),
        w_edge=0.7, w_var=0.3, gamma=0.7, floor=0.15, smooth=3,
        strength=mask_strength
    ).to(device)

    prewarm_renderer(H, W, k_sigma, device)

    # --- initial solution ---
    curr = new_individual(n_splats, H, W, min_scale_splats, max_scale_splats, device=device)
    curr_fit = fitness_population([curr], target, H, W, k_sigma, device,
                                  tile=32, chunk=None, weight_mask=imp_mask, boost_only=boost_only)[0]
    best = duplicate_individual(curr)
    best_fit = float(curr_fit)

    # curves (store energies = MSE)
    curves = {"best":[best_fit], "current":[float(curr_fit)]}

    # save first frame
    pad = len(str(iterations))
    if save_video and (0 % max(1, frame_every) == 0):
        save_frame_png(0, best, pad, prefix, video_dir, H, W, k_sigma, device, save_video)

    pbar = tqdm(range(iterations), desc="SA iterations", leave=True)
    try:
        for it in pbar:
            T = _temp_schedule(temp_schedule, temp0, it, iterations)
            # also cool mutation sigmas with your scheduler for stability
            # reuse your GA schedule helper via _anneal_factor
            _ = _anneal_factor(it, iterations, sigma_schedule)  # just for postfix progress

            accepted_any = False
            e_curr = float(curr_fit)

            for _try in range(tries_per_iter):
                # propose neighbor by GA-style mutation (single individual)
                neighbor = duplicate_individual(curr)
                neighbor = mutate_individual(
                    neighbor, is_elite=False, gen=it, total_gens=iterations,
                    schedule=sigma_schedule, mut_sigma_max=mut_sigma_max,
                    mut_sigma_min=mut_sigma_min, mutpb=mutpb,
                    H=H, W=W, min_scale_splats=min_scale_splats, max_scale_splats=max_scale_splats
                )

                e_new = fitness_population([neighbor], target, H, W, k_sigma, device,
                                           tile=32, chunk=None, weight_mask=imp_mask, boost_only=boost_only)[0]

                # ΔE (note: pseudocode used Δf with "higher is better"; here E=MSE, lower is better)
                dE = float(e_new) - e_curr
                if dE <= 0.0:
                    # improvement: accept
                    curr = neighbor
                    curr_fit = float(e_new)
                    e_curr = curr_fit
                    accepted_any = True
                else:
                    # worse: accept with probability exp(-ΔE/T)
                    if T > 0.0:
                        prob = math.exp(-dE / T)
                        if random.random() < prob:
                            curr = neighbor
                            curr_fit = float(e_new)
                            e_curr = curr_fit
                            accepted_any = True

                # update global best
                if e_curr + 1e-12 < best_fit:
                    best_fit = e_curr
                    best = duplicate_individual(curr)

            # curves/logs
            curves["best"].append(best_fit)
            curves["current"].append(float(curr_fit))

            # frames
            if save_video and ((it + 1) % max(1, frame_every) == 0):
                save_frame_png(it + 1, best, pad, prefix, video_dir, H, W, k_sigma, device, save_video)

            pbar.set_postfix(best_mse=f"{best_fit:.6f}", curr_mse=f"{float(curr_fit):.6f}",
                             T=f"{T:.4g}", accepted="Y" if accepted_any else "N")
    except KeyboardInterrupt:
        try:
            pbar.close()
        except Exception:
            pass
        print("\n[Interrupted] Returning current best…", flush=True)
    finally:
        try:
            pbar.close()
        except Exception:
            pass

    # save curves
    try:
        save_loss_curve_png(
            curves, loss_png_path,
            title=f"{prefix} energy (MSE)",
            xlabel="Iteration", ylabel="MSE",
            log_y=loss_log_y, dpi=144
        )
        save_curves_csv(curves, loss_csv_path)
        if loss_png_path:
            print(f"Saved loss plot to {loss_png_path}")
        if loss_csv_path:
            print(f"Saved loss CSV to {loss_csv_path}")
    except Exception as e:
        print(f"[warn] Could not save SA curves: {e}")

    return best.cpu(), float(best_fit)
