import torch


@torch.no_grad()
def axes_angle_to_cholesky(a_log: torch.Tensor, b_log: torch.Tensor, theta: torch.Tensor):
    """
    Given sigma_x=exp(a_log), sigma_y=exp(b_log) and rotation theta,
    compute the lower-triangular Cholesky L of Σ = R D R^T:
        L = [[l11, 0],
             [l21, l22]]
    Return a_log_eff=log(l11), b_log_eff=log(l22), c_raw_eff=l21.
    """
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
    """
    Convert a genome with theta to renderer format, **without alpha**.

    Input (new, preferred): [N,8]
        [x, y, a_log, b_log, theta, r, g, b]

    Also accepts legacy [N,9] with a trailing alpha column; alpha is ignored.

    Output (renderer format): [N,8]
        [x, y, a_log_eff, b_log_eff, c_raw_eff, r, g, b]
    """
    if ind_axes_angle.ndim == 1:
        ind_axes_angle = ind_axes_angle.unsqueeze(0)

    N, C = ind_axes_angle.shape

    out = torch.empty((N, 8), device=ind_axes_angle.device, dtype=ind_axes_angle.dtype)

    # Copy x, y and colors
    out[:, 0:2] = ind_axes_angle[:, 0:2]      # x, y
    out[:, 5:8] = ind_axes_angle[:, 5:8]      # r, g, b

    # Convert (a_log, b_log, theta) -> (a_log_eff, b_log_eff, c_raw_eff)
    a_log_eff, b_log_eff, c_raw_eff = axes_angle_to_cholesky(
        ind_axes_angle[:, 2],  # a_log
        ind_axes_angle[:, 3],  # b_log
        ind_axes_angle[:, 4],  # theta
    )
    out[:, 2] = a_log_eff
    out[:, 3] = b_log_eff
    out[:, 4] = c_raw_eff

    return out
