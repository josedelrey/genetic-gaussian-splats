# Configuration constants for Genetic Gaussian Splats

# Evolution parameters
DEFAULT_WORK_RESOLUTION = (128, 128)  # H, W
WORK_MAX_SIDE = 512
N_SPLATS = 512
POP_SIZE = 8
GENERATIONS = 500000
TOUR_K = 2  # Tournament selection size
ELITE_K = 4  # Number of elites to preserve
CXPB = 0.05  # Crossover probability

# Mutation parameters
MUTPB = 0.01  # Mutation probability
PROTECT_BEST_ELITE = True

# Rendering parameters
K_SIGMA = 3.0  # Gaussian extent multiplier
DEFAULT_TILE_SIZE = 32

# min_scale_splats: absolute minimum sigma in pixels
# max_scale_splats: maximum sigma as fraction of image size
MIN_SCALE_SPLATS = 3.0
MAX_SCALE_SPLATS = 0.1

# Mutation sigma schedules
MUT_SIGMA_MAX = {
    "xy": 0.1, 
    "alog": 0.5, 
    "blog": 0.5, 
    "theta": 0.3,
    "rgb": 25.0, 
    "alpha": 25.0
}

MUT_SIGMA_MIN = {
    "xy": 0.01,
    "alog": 0.05, 
    "blog": 0.05, 
    "theta": 0.025,
    "rgb": 2.0, 
    "alpha": 2.0
}

# Annealing schedule: "linear", "cosine", "exp"
SCHEDULE = "cosine"

# Importance mask parameters
MASK_STRENGTH = 0.7  # 1.0 = full edge focus, 0.0 = plain MSE
BOOST_ONLY = False  # If True, use mask to only boost important areas over plain MSE

# Random seed
SEED = 42

# Directories
INPUT_DIR = "imgs"
OUTPUT_DIR = "output"
REF_IMG = "reference.jpg"

# Video saving
SAVE_VIDEO = True
VIDEO_LEN = 10
FPS = 30
FRAME_EVERY = max(1, GENERATIONS // (FPS * VIDEO_LEN))

# Loss curve outputs
SAVE_LOSS_CURVE = True
LOSS_LOG_Y = True

# ----- SA parameters (Simulated Annealing) -----
SA_TRIES_PER_ITER = 1           # neighbors per iteration (can raise to 2~8)
SA_T0 = 5e-4                    # initial temperature (in MSE units). Tune per image/scale (0 = auto-estimate)
SA_SCHEDULE = "exp"             # "exp", "linear", "cosine", "log", "cauchy"
