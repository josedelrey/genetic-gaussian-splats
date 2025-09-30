import torch


@torch.no_grad()
def axes_angle_to_cholesky(a_log: torch.Tensor, b_log: torch.Tensor, theta: torch.Tensor):
    sigma_x = torch.exp(a_log)
    sigma_y = torch.exp(b_log)
    c = torch.cos(theta)
    s = torch.sin(theta)

    # Σ entries
    sxx = (sigma_x**2) * (c**2) + (sigma_y**2) * (s**2)
    sxy = (sigma_x**2 - sigma_y**2) * s * c
    syy = (sigma_x**2) * (s**2) + (sigma_y**2) * (c**2)

    eps = 1e-12
    l11 = torch.sqrt(torch.clamp(sxx, min=eps))
    l21 = sxy / l11
    l22 = torch.sqrt(torch.clamp(syy - l21*l21, min=eps))

    a_log_eff = torch.log(l11)
    b_log_eff = torch.log(l22)
    c_raw_eff = l21
    return a_log_eff, b_log_eff, c_raw_eff


@torch.no_grad()
def genome_to_renderer(ind_axes_angle: torch.Tensor) -> torch.Tensor:
    if ind_axes_angle.ndim == 1:
        ind_axes_angle = ind_axes_angle.unsqueeze(0)

    N, _ = ind_axes_angle.shape

    # Allocate output [N,9]
    out = torch.empty((N, 9), device=ind_axes_angle.device, dtype=ind_axes_angle.dtype)

    # xy
    out[:, 0:2] = ind_axes_angle[:, 0:2]

    # Convert (a_log,b_log,theta) to (a_log_eff,b_log_eff,c_raw_eff)
    a_log_eff, b_log_eff, c_raw_eff = axes_angle_to_cholesky(
        ind_axes_angle[:, 2],  # a_log
        ind_axes_angle[:, 3],  # b_log
        ind_axes_angle[:, 4],  # theta
    )
    out[:, 2] = a_log_eff
    out[:, 3] = b_log_eff
    out[:, 4] = c_raw_eff

    # Colors
    out[:, 5:8] = ind_axes_angle[:, 5:8]

    # Alpha
    out[:, 8] = ind_axes_angle[:, 8]

    # Clamp colors and alpha to [0,255]
    out[:, 5:9].clamp_(0.0, 255.0)

    return out


@torch.no_grad()
def genome_to_renderer_batched(G_axes: torch.Tensor) -> torch.Tensor:
    B, N, C = G_axes.shape
    Gf = G_axes.reshape(B * N, C)

    R = genome_to_renderer(Gf)
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
