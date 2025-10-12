from __future__ import annotations

import os
import numpy as np
import torch, random
from PIL import Image

from modules.render import render_splats_rgb_triton, _DEV as DEV
from modules.resize import choose_work_size, scale_genome_pixels_anisotropic
from modules.encode import genome_to_renderer
from modules.algorithm import genetic_approx
from modules.config import *


# Setup directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
VIDEO_DIR = os.path.join(OUTPUT_DIR, "video_frames")
if SAVE_VIDEO:
    os.makedirs(VIDEO_DIR, exist_ok=True)

# Loss outputs
LOSS_PNG = os.path.join(OUTPUT_DIR, "ga_loss.png") if SAVE_LOSS_CURVE else ""
LOSS_CSV = os.path.join(OUTPUT_DIR, "ga_loss.csv") if SAVE_LOSS_CURVE else ""

# Set random seed
if SEED is not None:
    random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


if __name__ == "__main__":
    img_path = os.path.join(INPUT_DIR, REF_IMG)
    pil_img = Image.open(img_path).convert("RGB")
    np_img = np.array(pil_img, dtype=np.float32) / 255.0
    target_img = torch.from_numpy(np_img)

    H_out, W_out = target_img.shape[0], target_img.shape[1]
    H, W = choose_work_size(H_out, W_out, max_side=WORK_MAX_SIDE)

    best_ind, best_fit = genetic_approx(
        target_img,
        H=H, W=W, device=DEV,
        # GA parameters
        pop_size=POP_SIZE, n_splats=N_SPLATS, generations=GENERATIONS,
        tour_k=TOUR_K, elite_k=ELITE_K, cxpb=CXPB, mutpb=MUTPB,
        mut_sigma_max=MUT_SIGMA_MAX, mut_sigma_min=MUT_SIGMA_MIN,
        schedule=SCHEDULE,
        min_scale_splats=MIN_SCALE_SPLATS, max_scale_splats=MAX_SCALE_SPLATS,
        k_sigma=K_SIGMA, mask_strength=MASK_STRENGTH, boost_only=BOOST_ONLY,
        # Video saving
        save_video=SAVE_VIDEO, frame_every=FRAME_EVERY,
        video_dir=VIDEO_DIR, prefix="ga",
        # Loss curves
        loss_png_path=LOSS_PNG, loss_csv_path=LOSS_CSV, loss_log_y=LOSS_LOG_Y
    )

    print("Best MSE:", best_fit)
    if LOSS_PNG:
        print(f"Loss plot saved to {LOSS_PNG}")
    if LOSS_CSV:
        print(f"Loss CSV saved to {LOSS_CSV}")

    if best_ind is not None:
        sH = H_out / float(H); sW = W_out / float(W)
        best_ind_full = scale_genome_pixels_anisotropic(best_ind.to(DEV), sH=sH, sW=sW)
        best_ind_full_render = genome_to_renderer(best_ind_full)

        final = render_splats_rgb_triton(
            best_ind_full_render.unsqueeze(0), H_out, W_out,
            k_sigma=K_SIGMA, device=DEV, tile=DEFAULT_TILE_SIZE
        )[0]
        img8 = (final.clamp(0, 1).detach().cpu().numpy() * 255).astype('uint8')
        out_path = os.path.join(OUTPUT_DIR, "ga_splats.png")

        Image.fromarray(img8).save(out_path)
        print("Saved full resolution result as ga_splats.png")

        if SAVE_VIDEO:
            print(f"Saved frames to {VIDEO_DIR}")
