# ===== File: ./run_sa.py =====
from __future__ import annotations
import os, random
import numpy as np
import torch
from PIL import Image

from modules.render import render_splats_rgb_triton, _DEV as DEV
from modules.resize import choose_work_size, scale_genome_pixels_anisotropic
from modules.encode import genome_to_renderer
from modules.annealing import simulated_annealing
from modules.config import *

# Setup dirs
os.makedirs(OUTPUT_DIR, exist_ok=True)
VIDEO_DIR = os.path.join(OUTPUT_DIR, "video_frames_sa")
if SAVE_VIDEO:
    os.makedirs(VIDEO_DIR, exist_ok=True)

# Loss outputs
SA_LOSS_PNG = os.path.join(OUTPUT_DIR, "sa_loss.png") if SAVE_LOSS_CURVE else ""
SA_LOSS_CSV = os.path.join(OUTPUT_DIR, "sa_loss.csv") if SAVE_LOSS_CURVE else ""

# Seed
if SEED is not None:
    random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

if __name__ == "__main__":
    # load target
    img_path = os.path.join(INPUT_DIR, REF_IMG)
    pil_img = Image.open(img_path).convert("RGB")
    np_img = np.array(pil_img, dtype=np.float32) / 255.0
    target_img = torch.from_numpy(np_img)

    # work size
    H_out, W_out = target_img.shape[0], target_img.shape[1]
    H, W = choose_work_size(H_out, W_out, max_side=WORK_MAX_SIDE)

    # run SA
    best_ind, best_fit = simulated_annealing(
        target_img, H=H, W=W, device=DEV,
        # genome / neighborhood
        n_splats=N_SPLATS,
        mutpb=MUTPB,
        mut_sigma_max=MUT_SIGMA_MAX,
        mut_sigma_min=MUT_SIGMA_MIN,
        sigma_schedule=SCHEDULE,  # reuse your GA schedule for mutation size
        min_scale_splats=MIN_SCALE_SPLATS,
        max_scale_splats=MAX_SCALE_SPLATS,

        # renderer / fitness
        k_sigma=K_SIGMA,
        mask_strength=MASK_STRENGTH,
        boost_only=BOOST_ONLY,

        # SA loop
        iterations=GENERATIONS,
        temp0=SA_T0,
        temp_schedule=SA_SCHEDULE,
        tries_per_iter=SA_TRIES_PER_ITER,

        # saving
        save_video=SAVE_VIDEO,
        frame_every=FRAME_EVERY,
        video_dir=VIDEO_DIR,
        prefix="sa",
        loss_png_path=SA_LOSS_PNG,
        loss_csv_path=SA_LOSS_CSV,
        loss_log_y=LOSS_LOG_Y
    )

    print("SA Best MSE:", best_fit)
    if SA_LOSS_PNG: print(f"Loss plot saved to {SA_LOSS_PNG}")
    if SA_LOSS_CSV: print(f"Loss CSV saved to {SA_LOSS_CSV}")

    # upscale best to full res + final render
    if best_ind is not None:
        sH = H_out / float(H); sW = W_out / float(W)
        best_ind_full = scale_genome_pixels_anisotropic(best_ind.to(DEV), sH=sH, sW=sW)
        best_ind_full_render = genome_to_renderer(best_ind_full)

        final = render_splats_rgb_triton(
            best_ind_full_render.unsqueeze(0), H_out, W_out,
            k_sigma=K_SIGMA, device=DEV, tile=DEFAULT_TILE_SIZE
        )[0]
        img8 = (final.clamp(0,1).detach().cpu().numpy() * 255).astype("uint8")
        out_path = os.path.join(OUTPUT_DIR, "sa_splats.png")
        Image.fromarray(img8).save(out_path)
        print("Saved full-resolution SA result as sa_splats.png")

        if SAVE_VIDEO:
            print(f"Saved SA frames to {VIDEO_DIR}")
