import torch
import random
from typing import List
from modules.population import duplicate_individual
from modules.utils import wrap_angle, build_mut_sigma, clamp_genome


def tournament_selection(pop: List[torch.Tensor], fits: List[float], k: int = 2) -> torch.Tensor:
    best_idx = None
    for _ in range(k):
        i = random.randrange(len(pop))
        if best_idx is None or fits[i] < fits[best_idx]:
            best_idx = i
    return duplicate_individual(pop[best_idx])


def crossover_uniform(a: torch.Tensor, b: torch.Tensor, p: float = 0.5):
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


def mutate_individual(ind: torch.Tensor, is_elite: bool, gen: int, total_gens: int, 
                     schedule: str, mut_sigma_max: dict, mut_sigma_min: dict,
                     mutpb: float, H: int, W: int, min_scale_splats: float, 
                     max_scale_splats: float):
    SIG = build_mut_sigma(gen, total_gens, schedule, mut_sigma_max, mut_sigma_min)

    with torch.no_grad():
        N = ind.shape[0]

        # Masks for which genes to mutate
        m_xy = (torch.rand((N, 2), device=ind.device) < mutpb)
        m_ab = (torch.rand((N, 2), device=ind.device) < mutpb)
        m_t  = (torch.rand((N, 1), device=ind.device) < mutpb)

        # Color + alpha mutation, ensure at least one of RGBA mutates
        m_rgb_flag = (torch.rand((N, 1), device=ind.device) < mutpb)
        m_a_flag   = (torch.rand((N, 1), device=ind.device) < mutpb)

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

        clamp_genome(ind, H, W, min_scale_splats, max_scale_splats)

        # Swap two splats
        if N >= 2:
            i = int(torch.randint(0, N - 1, (1,), device=ind.device).item())
            size = torch.exp(ind[:, 2]) * torch.exp(ind[:, 3])
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