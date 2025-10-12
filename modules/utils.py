import math
import torch
import numpy as np
import os
import csv
from PIL import Image
from typing import Dict, Sequence


@torch.no_grad()
def wrap_angle(theta: torch.Tensor) -> torch.Tensor:
    """Wrap angle to [-π, π] range."""
    return (theta + np.pi) % (2*np.pi) - np.pi


def _anneal_factor(gen, total, kind):
    """Compute annealing factor for mutation schedules."""
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


def build_mut_sigma(gen: int, total_gens: int, kind: str, mut_sigma_max: dict, mut_sigma_min: dict):
    """Build mutation sigma values for current generation."""
    f = _anneal_factor(gen, total_gens, kind)
    return {k: mut_sigma_min[k] + f * (mut_sigma_max[k] - mut_sigma_min[k]) for k in mut_sigma_max.keys()}


def clamp_genome(ind: torch.Tensor, H: int, W: int, min_scale_splats: float, max_scale_splats: float) -> torch.Tensor:
    """Clamp genome parameters to valid ranges."""
    ind[:,0:2] = ind[:,0:2].clamp(0.0, 1.0)
    max_side = float(max(H, W))
    min_scale_log = torch.log(torch.tensor(min_scale_splats, device=ind.device))
    max_scale_log = torch.log(torch.tensor(max_scale_splats * max_side, device=ind.device))
    ind[:,2] = ind[:,2].clamp(min_scale_log, max_scale_log)
    ind[:,3] = ind[:,3].clamp(min_scale_log, max_scale_log)
    ind[:,4] = wrap_angle(ind[:,4])
    ind[:,5:9] = ind[:,5:9].clamp(0.0, 255.0)
    return ind


@torch.no_grad()
def render_axes_angle_to_img(ind_axes_angle: torch.Tensor, Hsnap: int, Wsnap: int, 
                           k_sigma: float, device) -> np.ndarray:
    """Render individual genome to numpy image."""
    from modules.encode import genome_to_renderer_batched
    from modules.render import render_splats_rgb_triton
    
    G = ind_axes_angle.unsqueeze(0) if ind_axes_angle.ndim == 2 else ind_axes_angle  # [1,N,C]
    G9 = genome_to_renderer_batched(G)  # [1,N,9]
    img = render_splats_rgb_triton(G9, Hsnap, Wsnap, k_sigma=k_sigma, device=device, tile=32)[0]  # [H,W,3]
    img8 = (img.clamp(0,1).detach().cpu().numpy() * 255.0).astype("uint8")
    return img8


@torch.no_grad()
def save_frame_png(gen: int, ind_axes_angle: torch.Tensor, pad: int, prefix: str, 
                  video_dir: str, H: int, W: int, k_sigma: float, device, 
                  save_video: bool = True):
    """Save frame to PNG file."""
    if not save_video: 
        return
    img8 = render_axes_angle_to_img(ind_axes_angle, H, W, k_sigma, device)
    fname = f"{prefix}_{gen:0{pad}d}.png"
    Image.fromarray(img8).save(os.path.join(video_dir, fname))


@torch.no_grad()
def prewarm_renderer(H: int, W: int, k_sigma: float, device):
    """Prewarm the renderer to trigger CUDA context and Triton JIT."""
    from modules.render import render_splats_rgb_triton
    
    dummy = torch.tensor([[[0.5,0.5, math.log(2.0), math.log(2.0), 0.0, 128.0,128.0,128.0, 255.0]]],
                         device=device, dtype=torch.float32)  # [1,1,9]
    _ = render_splats_rgb_triton(dummy, min(8,H), min(8,W), k_sigma=k_sigma, device=device, tile=32)
    _ = render_splats_rgb_triton(dummy, min(8,H), min(8,W), k_sigma=k_sigma, device=device, tile=32)
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def save_loss_curve_png(curves: Dict[str, Sequence[float]],
                         out_path: str,
                         title: str = "GA fitness over generations",
                         xlabel: str = "Generation",
                         ylabel: str = "MSE",
                         log_y: bool = False,
                         dpi: int = 144) -> None:
    """
    Simple Matplotlib plot of one or more curves. Keys are labels.
    """
    if not out_path:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[warn] matplotlib not available, cannot save plot: {e}")
        return

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # All curves must have the same length
    Ls = [len(v) for v in curves.values() if len(v) > 0]
    if not Ls:
        print("[warn] No values to plot")
        return
    L = Ls[0]
    for k, v in curves.items():
        if len(v) != L:
            raise ValueError(f"Curve '{k}' length {len(v)} does not match others {L}")

    gens = list(range(L))
    plt.figure()
    for name, values in curves.items():
        if len(values) == 0:
            continue
        plt.plot(gens, values, label=name)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if log_y:
        plt.yscale("log")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close()


def save_curves_csv(curves: Dict[str, Sequence[float]], out_csv_path: str) -> None:
    """
    Save curves to CSV with header: gen,<key1>,<key2>,...
    """
    if not out_csv_path:
        return
    os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)
    keys = list(curves.keys())
    Ls = [len(v) for v in curves.values() if len(v) > 0]
    if not Ls:
        print("[warn] No values to save to CSV")
        return
    L = Ls[0]
    with open(out_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["gen"] + keys)
        for i in range(L):
            row = [i] + [curves[k][i] if i < len(curves[k]) else "" for k in keys]
            writer.writerow(row)
