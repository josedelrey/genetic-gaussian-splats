import torch
from typing import List
from modules.encode import genome_to_renderer_batched
from modules.render import render_splats_rgb_triton


@torch.no_grad()
def fitness_many(pop_batch: List[torch.Tensor], target: torch.Tensor, H: int, W: int,
                k_sigma: float, device, tile: int = 32,
                weight_mask: torch.Tensor | None = None,
                boost_only: bool = False,
                boost_beta: float = 1.0):
    G_axes = torch.stack(pop_batch, dim=0)  # [B,N,C]
    G9 = genome_to_renderer_batched(G_axes)  # [B,N,9]
    imgs = render_splats_rgb_triton(G9, H, W, k_sigma=k_sigma, device=device, tile=tile)  # [B,H,W,3]
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
                      H: int, W: int, k_sigma: float, device,
                      tile: int = 32, chunk: int | None = None,
                      weight_mask: torch.Tensor | None = None,
                      boost_only: bool = False) -> List[float]:
    if chunk is None or chunk >= len(population):
        return fitness_many(population, target, H, W, k_sigma, device, tile=tile, 
                            weight_mask=weight_mask, boost_only=boost_only).detach().cpu().tolist()
    
    out: List[float] = []
    for i in range(0, len(population), chunk):
        out.extend(fitness_many(population[i:i+chunk], target, H, W, k_sigma, device, tile=tile, 
                                weight_mask=weight_mask, boost_only=boost_only).detach().cpu().tolist())
    return out