import math
import torch
from typing import List


def sample_log_scales_beta_linear(B, N, s_lo, s_hi, m=0.5, concentration=8.0, device='cuda', dtype=torch.float32):
    # m in [0,1] is where you want the mean in the *linear* [s_lo, s_hi] range.
    eps = 1e-6
    alpha = torch.tensor(m * max(concentration, eps) + eps, device=device, dtype=dtype)
    beta  = torch.tensor((1 - m) * max(concentration, eps) + eps, device=device, dtype=dtype)

    dist = torch.distributions.Beta(alpha, beta)
    u = dist.sample((B, N, 1))                       # in [0,1]
    sigma = s_lo + u * (s_hi - s_lo)                 # map to [s_lo, s_hi]
    log_sigma = sigma.log()
    return log_sigma  # shape (B, N, 1)



@torch.no_grad()
def new_population(batch_size: int, n_splats: int, H: int, W: int, 
                  min_scale_splats: float, max_scale_splats: float,
                  device='cuda', dtype=torch.float32) -> torch.Tensor:
    """Create new random population of genomes."""
    B, N = batch_size, n_splats
    max_side = float(max(H, W))

    # x, y in [0,1]
    xy = torch.empty(B, N, 2, device=device, dtype=dtype).uniform_(0.0, 1.0)

    # Scales (sample in linear-sigma, then log)
    s_lo = float(min_scale_splats)
    s_hi = float(max_scale_splats * max_side)
    a = sample_log_scales_beta_linear(B, N, s_lo, s_hi, m=0.3, concentration=7.0, device=device, dtype=dtype)
    b = sample_log_scales_beta_linear(B, N, s_lo, s_hi, m=0.5, concentration=7.0, device=device, dtype=dtype)


    # Theta
    theta = torch.empty(B, N, 1, device=device, dtype=dtype).uniform_(-math.pi, math.pi)

    # Colors + alpha
    rgb   = torch.empty(B, N, 3, device=device, dtype=dtype).uniform_(0.0, 256.0)
    alpha = torch.empty(B, N, 1, device=device, dtype=dtype).uniform_(180.0, 256.0)

    G = torch.cat([xy, a, b, theta, rgb, alpha], dim=-1)  # [B,N,9]
    G[..., 0:2].clamp_(0.0, 1.0)
    G[..., 5:9].clamp_(0.0, 255.0)
    return G


def new_individual(n_splats: int, H: int, W: int, min_scale_splats: float, 
                  max_scale_splats: float, device='cuda') -> torch.Tensor:
    """Create new random individual genome."""
    return new_population(1, n_splats, H, W, min_scale_splats, max_scale_splats, device=device)[0]


def duplicate_individual(ind: torch.Tensor) -> torch.Tensor:
    """Create a deep copy of an individual."""
    return ind.clone()


def population_to_list(pop_tensor: torch.Tensor) -> List[torch.Tensor]:
    """Convert population tensor to list of individuals."""
    return [pop_tensor[i] for i in range(pop_tensor.shape[0])]