# tools/interactive_splat.py
from __future__ import annotations
import os, math, sys, numpy as np, torch
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, CheckButtons

# Use your renderer
from modules.render_triton import render_splats_rgb_triton, _DEV as DEV
from modules.encode import genome_to_renderer


# ----------------------------
# Config (match your GA)
# ----------------------------
H = 512
W = 512
K_SIGMA = 3.0
TILE_DEFAULT = 64
BACKGROUND = (1.0, 1.0, 1.0)

# ----------------------------
# Helpers
# ----------------------------
@torch.no_grad()
def prewarm_renderer(H:int, W:int, device=DEV):
    """Trigger CUDA context + Triton JIT once, on tiny shapes, and cache variants."""
    dummy = torch.tensor([[[0.5,0.5, math.log(3.0), math.log(3.0), 0.0, 128.0,128.0,128.0, 255.0]]],
                         device=device, dtype=torch.float32)  # [1,1,9]
    _ = render_splats_rgb_triton(dummy, min(8,H), min(8,W), k_sigma=K_SIGMA, device=device, tile=32)
    _ = render_splats_rgb_triton(dummy, min(8,H), min(8,W), k_sigma=K_SIGMA, device=device, tile=64)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def wrap_angle(theta: torch.Tensor | float) -> float:
    th = float(theta)
    return (th + math.pi) % (2 * math.pi) - math.pi

def build_genome(x, y, sig_a_pix, sig_b_pix, theta, r, g, b, a255, device=DEV):
    # Build axes+theta genome: [N,8] = [x,y, a_log, b_log, theta, r,g,b]
    a_log = math.log(max(1e-3, sig_a_pix))
    b_log = math.log(max(1e-3, sig_b_pix))
    theta = wrap_angle(theta)

    axes = torch.tensor([[x, y, a_log, b_log, theta, r, g, b]],
                        device=device, dtype=torch.float32)          # [1,8]
    rend = genome_to_renderer(axes)                                   # [1,8] -> [x,y,a_log_eff,b_log_eff,c_raw_eff,r,g,b]
    alpha = torch.tensor([[a255]], device=device, dtype=torch.float32)  # [1,1]
    full = torch.cat([rend, alpha], dim=1)                             # [1,9]
    return full.unsqueeze(0)                                           # [1,1,9]


@torch.no_grad()
def render_one(genome, H, W, tile=TILE_DEFAULT, bg=BACKGROUND):
    img = render_splats_rgb_triton(genome, H, W, k_sigma=K_SIGMA, device=DEV, tile=tile, background=bg)[0]
    return (img.clamp(0,1).detach().cpu().numpy() * 255).astype("uint8")

# ----------------------------
# Init
# ----------------------------
device = DEV
assert torch.device(device).type == "cuda", "This interactive viewer needs CUDA."

prewarm_renderer(H, W, device=device)

# Initial values
init = dict(
    x=0.5, y=0.5,
    sig_a=24.0,  # pixels
    sig_b=12.0,  # pixels
    theta=0.0,
    r=200.0, g=80.0, b=40.0,
    a=200.0,    # alpha 0..255 (used as blend weight)
    tile=TILE_DEFAULT
)

genome = build_genome(init["x"], init["y"], init["sig_a"], init["sig_b"], init["theta"],
                      init["r"], init["g"], init["b"], init["a"], device=device)
img = render_one(genome, H, W, tile=init["tile"])

# ----------------------------
# UI layout
# ----------------------------
matplotlib.use('Qt5Agg' if 'Qt' in matplotlib.rcsetup.all_backends else matplotlib.get_backend())
# --- Layout: main image left, controls right ---
fig = plt.figure(figsize=(11, 7))

# Image takes ~70% width
ax_img = plt.axes([0.05, 0.05, 0.65, 0.90])
ax_img.set_title("Interactive Gaussian Splat (CUDA + Triton)")
im = ax_img.imshow(img)
ax_img.axis("off")

# Right-side panel for sliders
slider_left = 0.75
slider_width = 0.20
slider_height = 0.03
slider_pad = 0.01

s_x  = Slider(plt.axes([slider_left, 0.90, slider_width, slider_height]), 'x', 0.0, 1.0, valinit=init["x"])
s_y  = Slider(plt.axes([slider_left, 0.90 - 1*(slider_height+slider_pad), slider_width, slider_height]), 'y', 0.0, 1.0, valinit=init["y"])
s_sa = Slider(plt.axes([slider_left, 0.90 - 2*(slider_height+slider_pad), slider_width, slider_height]), 'σₐ', 1.0, 200.0, valinit=init["sig_a"])
s_sb = Slider(plt.axes([slider_left, 0.90 - 3*(slider_height+slider_pad), slider_width, slider_height]), 'σ_b', 1.0, 200.0, valinit=init["sig_b"])
s_th = Slider(plt.axes([slider_left, 0.90 - 4*(slider_height+slider_pad), slider_width, slider_height]), 'θ', -math.pi, math.pi, valinit=init["theta"])
s_r  = Slider(plt.axes([slider_left, 0.90 - 5*(slider_height+slider_pad), slider_width, slider_height]), 'R', 0.0, 255.0, valinit=init["r"])
s_g  = Slider(plt.axes([slider_left, 0.90 - 6*(slider_height+slider_pad), slider_width, slider_height]), 'G', 0.0, 255.0, valinit=init["g"])
s_b_ = Slider(plt.axes([slider_left, 0.90 - 7*(slider_height+slider_pad), slider_width, slider_height]), 'B', 0.0, 255.0, valinit=init["b"])
s_a  = Slider(plt.axes([slider_left, 0.90 - 8*(slider_height+slider_pad), slider_width, slider_height]), 'α', 0.0, 255.0, valinit=init["a"])

# Buttons under sliders
bx_rand  = plt.axes([slider_left, 0.05, 0.09, 0.05])
bx_reset = plt.axes([slider_left+0.11, 0.05, 0.09, 0.05])
b_rand   = Button(bx_rand, 'Randomize')
b_reset  = Button(bx_reset, 'Reset')

# Checkboxes below buttons
cx = plt.axes([slider_left, 0.15, 0.20, 0.10], facecolor='lightgoldenrodyellow')
checks = CheckButtons(cx, ['Checker BG', 'Tile=32'], [False, False])


# ----------------------------
# Events
# ----------------------------
checker_bg = False
tile32 = False

def current_bg():
    if checker_bg:
        # simple light checker pattern (done on CPU then sent as background by overwriting image after render)
        # but since renderer accepts solid bg, we emulate by blending after render
        # Instead: keep renderer white and compose checker separately here:
        return (1.0, 1.0, 1.0)
    return BACKGROUND

def draw():
    global genome
    # Build genome from sliders
    x = float(s_x.val)
    y = float(s_y.val)
    sa = float(s_sa.val)
    sb = float(s_sb.val)
    th = float(s_th.val)
    r = float(s_r.val)
    g = float(s_g.val)
    b = float(s_b_.val)
    a = float(s_a.val)

    genome = build_genome(x, y, sa, sb, th, r, g, b, a, device=device)
    tile = 32 if tile32 else TILE_DEFAULT

    out = render_one(genome, H, W, tile=tile, bg=current_bg())

    if checker_bg:
        # Compose over a checkerboard to visualize transparency-ish effect
        # (renderer already uses alpha as weight, not premultiplied alpha)
        # We'll just blend the rendered colors over a 2D checker produced here for visualization.
        sz = 32
        yy, xx = np.mgrid[0:H, 0:W]
        chk = ((xx // sz + yy // sz) % 2).astype(np.float32)
        # light gray / white
        c0 = 220.0
        c1 = 255.0
        bg = (chk[..., None] * (c1 - c0) + c0).astype(np.uint8)
        # Simple max to keep splat visible (visual, not physically correct alpha)
        out = np.maximum(out, bg)

    im.set_data(out)
    fig.canvas.draw_idle()

def on_slider_change(val):
    draw()

def on_rand(event):
    import random
    s_x.set_val(np.clip(np.random.rand(), 0, 1))
    s_y.set_val(np.clip(np.random.rand(), 0, 1))
    s_sa.set_val(np.random.uniform(4.0, 120.0))
    s_sb.set_val(np.random.uniform(4.0, 120.0))
    s_th.set_val(np.random.uniform(-math.pi, math.pi))
    s_r.set_val(np.random.uniform(0, 255))
    s_g.set_val(np.random.uniform(0, 255))
    s_b_.set_val(np.random.uniform(0, 255))
    s_a.set_val(np.random.uniform(32, 255))

def on_reset(event):
    s_x.set_val(0.5)
    s_y.set_val(0.5)
    s_sa.set_val(24.0)
    s_sb.set_val(12.0)
    s_th.set_val(0.0)
    s_r.set_val(200.0)
    s_g.set_val(80.0)
    s_b_.set_val(40.0)
    s_a.set_val(200.0)

def on_checks(label):
    global checker_bg, tile32
    if label == 'Checker BG':
        checker_bg = not checker_bg
    elif label == 'Tile=32':
        tile32 = not tile32
    draw()

def on_click(event):
    # Mouse click to set center (axes coords)
    if event.inaxes != ax_img: return
    if event.xdata is None or event.ydata is None: return
    x = float(event.xdata) / float(W)
    y = float(event.ydata) / float(H)
    s_x.set_val(np.clip(x, 0.0, 1.0))
    s_y.set_val(np.clip(y, 0.0, 1.0))

# Wire events
for s in [s_x, s_y, s_sa, s_sb, s_th, s_r, s_g, s_b_, s_a]:
    s.on_changed(on_slider_change)
b_rand.on_clicked(on_rand)
b_reset.on_clicked(on_reset)
checks.on_clicked(on_checks)
fig.canvas.mpl_connect('button_press_event', on_click)

# First draw
draw()
plt.show()
