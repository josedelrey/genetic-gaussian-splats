import torch
import torch.nn.functional as F
import random
from typing import Tuple
from tqdm.auto import tqdm
from statistics import median

from modules.population import new_population, duplicate_individual, population_to_list
from modules.fitness import fitness_population
from modules.genetic import tournament_selection, crossover_uniform, mutate_individual
from modules.mask import compute_importance_mask
from modules.utils import prewarm_renderer, save_frame_png, _anneal_factor
from modules.utils import save_loss_curve_png, save_curves_csv


@torch.no_grad()
def genetic_approx(target_img_uint8: torch.Tensor,
                   H: int, W: int, device,
                   # GA parameters
                   pop_size: int, n_splats: int, generations: int,
                   tour_k: int, elite_k: int, cxpb: float, mutpb: float,
                   mut_sigma_max: dict, mut_sigma_min: dict, schedule: str,
                   min_scale_splats: float, max_scale_splats: float,
                   k_sigma: float, mask_strength: float, boost_only: bool,
                   # Video saving
                   save_video: bool = False, frame_every: int = 5000,
                   video_dir: str = "", prefix: str = "ga",
                   # Loss curve outputs
                   loss_png_path: str = "",
                   loss_csv_path: str = "",
                   loss_log_y: bool = False) -> Tuple[torch.Tensor, float]:
    """
    Main genetic algorithm loop.
    Also records fitness curves and optionally saves a PNG and CSV at the end.
    """

    # Prepare target image
    t = target_img_uint8.to(torch.float32)
    if t.max() > 1.5:
        t = t / 255.0
    if t.shape[0] != H or t.shape[1] != W:
        tBCHW = t.permute(2, 0, 1).unsqueeze(0)
        t = F.interpolate(tBCHW, size=(H, W), mode='bilinear', align_corners=False)[0].permute(1, 2, 0)
    t = t.contiguous()

    # Compute importance mask
    imp_mask = compute_importance_mask(
        t, H, W,
        edge_scales=(1, 2, 4),
        w_edge=0.7, w_var=0.3,
        gamma=0.7, floor=0.15,
        smooth=3,
        strength=mask_strength
    ).to(device)

    target = t.to(device)
    prewarm_renderer(H, W, k_sigma, device)

    # Initialize population
    pop_tensor = new_population(pop_size, n_splats, H, W, min_scale_splats, max_scale_splats, device=device)
    population = population_to_list(pop_tensor)

    # Initial fitness evaluation
    fitnesses = fitness_population(
        population, target, H, W, k_sigma, device,
        tile=32, chunk=None, weight_mask=imp_mask, boost_only=boost_only
    )

    # Track best individual
    best_idx = min(range(pop_size), key=lambda i: fitnesses[i])
    best_ind = duplicate_individual(population[best_idx])
    best_fit = fitnesses[best_idx]
    no_improve = 0

    # Curves
    curves = {
        "best": [float(best_fit)],
        "mean": [float(sum(fitnesses) / len(fitnesses))],
        "median": [float(median(fitnesses))]
    }

    # Save initial frame
    pad = len(str(generations))
    if save_video and (0 % max(1, frame_every) == 0):
        save_frame_png(0, best_ind, pad, prefix, video_dir, H, W, k_sigma, device, save_video)

    # Main evolution loop
    pbar = tqdm(range(1, generations + 1), desc="GA generations", leave=True)
    try:
        for gen in pbar:
            # Parent selection
            parents = []
            while len(parents) < pop_size:
                parents.append(tournament_selection(population, fitnesses, k=tour_k))
            random.shuffle(parents)

            # Create offspring
            offspring = []
            for i in range(0, pop_size, 2):
                a = parents[i]
                b = parents[(i + 1) % pop_size]
                if random.random() < cxpb:
                    c1, c2 = crossover_uniform(a, b)
                else:
                    c1, c2 = duplicate_individual(a), duplicate_individual(b)

                # Mutate offspring
                c1_mut = mutate_individual(
                    c1, is_elite=False, gen=gen, total_gens=generations,
                    schedule=schedule, mut_sigma_max=mut_sigma_max,
                    mut_sigma_min=mut_sigma_min, mutpb=mutpb,
                    H=H, W=W, min_scale_splats=min_scale_splats,
                    max_scale_splats=max_scale_splats
                )
                offspring.append(c1_mut)

                if len(offspring) < pop_size:
                    c2_mut = mutate_individual(
                        c2, is_elite=False, gen=gen, total_gens=generations,
                        schedule=schedule, mut_sigma_max=mut_sigma_max,
                        mut_sigma_min=mut_sigma_min, mutpb=mutpb,
                        H=H, W=W, min_scale_splats=min_scale_splats,
                        max_scale_splats=max_scale_splats
                    )
                    offspring.append(c2_mut)

            # Evaluate offspring fitness
            off_fits = fitness_population(
                offspring, target, H, W, k_sigma, device,
                tile=32, chunk=None, weight_mask=imp_mask, boost_only=boost_only
            )

            # Elite selection
            elite_k_actual = max(1, elite_k)
            elite_idx = sorted(range(pop_size), key=lambda i: fitnesses[i])[:elite_k_actual]
            elites = [duplicate_individual(population[i]) for i in elite_idx]

            # Evaluate elite fitness
            elite_fits = fitness_population(
                elites, target, H, W, k_sigma, device,
                tile=32, chunk=None, weight_mask=imp_mask, boost_only=boost_only
            )

            # Form next generation
            population = elites + offspring[:pop_size - elite_k_actual]
            fitnesses = elite_fits + off_fits[:pop_size - elite_k_actual]

            # Update best individual
            gbest_idx = min(range(pop_size), key=lambda i: fitnesses[i])
            if fitnesses[gbest_idx] + 1e-10 < best_fit:
                best_fit = fitnesses[gbest_idx]
                best_ind = duplicate_individual(population[gbest_idx])
                no_improve = 0
            else:
                no_improve += 1

            # Log curves for this generation
            curves["best"].append(float(best_fit))
            curves["mean"].append(float(sum(fitnesses) / len(fitnesses)))
            curves["median"].append(float(median(fitnesses)))

            # Save frame
            if save_video and (gen % max(1, frame_every) == 0):
                save_frame_png(gen, best_ind, pad, prefix, video_dir, H, W, k_sigma, device, save_video)

            # Update progress bar
            f = _anneal_factor(gen, generations, schedule)
            pbar.set_postfix(best_mse=f"{best_fit:.6f}", stale=no_improve, sigma_fac=f"{f:.3f}")

    except KeyboardInterrupt:
        try:
            pbar.close()
        except Exception:
            pass
        print("\n[Interrupted] Returning current best individual…", flush=True)
    finally:
        try:
            pbar.close()
        except Exception:
            pass

    # Save curves
    try:
        save_loss_curve_png(
            curves, loss_png_path,
            title=f"{prefix} fitness",
            xlabel="Generation",
            ylabel="MSE",
            log_y=loss_log_y,
            dpi=144
        )
        save_curves_csv(curves, loss_csv_path)
        if loss_png_path:
            print(f"Saved loss plot to {loss_png_path}")
        if loss_csv_path:
            print(f"Saved loss CSV to {loss_csv_path}")
    except Exception as e:
        print(f"[warn] Could not save loss curves: {e}")

    return best_ind.cpu(), best_fit
