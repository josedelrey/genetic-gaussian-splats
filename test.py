# examples/smoke_test.py
import torch
from PIL import Image
import numpy as np

from modules.render import render_splats_rgb

def main():
    H, W = 128, 128

    # Two splats. Note: a_log/b_log are *logs* of pixel scales; c_raw is signed (pixels).
    genome = torch.tensor([
        [
            0.5, 0.5,                         # x, y (normalized)
            torch.log(torch.tensor(20.0)),    # a_log  -> l11 ≈ 20 px
            torch.log(torch.tensor(8.0)),     # b_log  -> l22 ≈  8 px
            6.0,                               # c_raw  -> shear (tilt)
            60.0, 160.0, 240.0,               # r, g, b
            0.9                                # alpha
        ],
        [
            0.30, 0.70,
            torch.log(torch.tensor(10.0)),
            torch.log(torch.tensor(10.0)),
            0.0,
            240.0, 80.0, 80.0,
            0.6
        ]
    ], dtype=torch.float32)

    img = render_splats_rgb(genome, H, W, k_sigma=3.0)
    arr = (img.numpy() * 255).astype(np.uint8)
    Image.fromarray(arr).save("out.png")
    print("Wrote out.png; shape:", arr.shape, "min/max:", arr.min(), arr.max())

if __name__ == "__main__":
    main()
