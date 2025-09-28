# test_render_inits.py
from __future__ import annotations
import os
import numpy as np
from PIL import Image
import torch

# --- import your GA module (rename 'main' if your file has another name)
from run_ggs import (
    new_individual,
    H, W, WORK_MAX_SIDE, N_SPLATS, K_SIGMA,  # read settings
)
# We’ll also import the module itself to update H,W in its namespace
import run_ggs as ga

from modules.render import render_splats_rgb, _DEV as DEV
from modules.resize import choose_work_size
from modules.encode import genome_to_renderer

OUT_DIR = "tests_inits"
IMG_PATH = os.path.join("imgs", "dog.jpg")  # same as your example file

os.makedirs(OUT_DIR, exist_ok=True)

def render_and_save(ind_axes_angle: torch.Tensor, h: int, w: int, idx: int):
    """Convert to renderer format, render, and save PNG."""
    ind_render = genome_to_renderer(ind_axes_angle).to(DEV)
    img = render_splats_rgb(ind_render, h, w, k_sigma=K_SIGMA, device=DEV)  # [H,W,3] in [0,1]
    img8 = (img.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(img8).save(os.path.join(OUT_DIR, f"init_{idx:02d}.png"))

def main():
    # --- Match the working canvas exactly like your main script
    if not os.path.isfile(IMG_PATH):
        raise FileNotFoundError(f"Cannot find {IMG_PATH}. Put a test image there or edit IMG_PATH.")

    pil_img = Image.open(IMG_PATH).convert("RGB")
    np_img = np.array(pil_img, dtype=np.float32) / 255.0
    target_img = torch.from_numpy(np_img)  # [H_out, W_out, 3]

    H_out, W_out = target_img.shape[0], target_img.shape[1]
    H_work, W_work = choose_work_size(H_out, W_out, max_side=WORK_MAX_SIDE)

    # --- IMPORTANT: update H, W inside the GA module so its helpers clamp consistently
    ga.H, ga.W = H_work, W_work

    print(f"Working canvas: H={H_work}, W={W_work}  |  N_SPLATS={N_SPLATS}  |  K_SIGMA={K_SIGMA}")
    print(f"Saving renders to: {OUT_DIR}")

    # --- Generate and render 10 different fully-initialized individuals
    for i in range(10):
        indiv = new_individual(N_SPLATS, device=DEV)  # [N_SPLATS, 9] (axes+angle genome)
        render_and_save(indiv, H_work, W_work, i)

    print("Done. Open the tests_inits/ folder to inspect the initializations.")

if __name__ == "__main__":
    main()
