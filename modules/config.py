# Configuration constants for Genetic Gaussian Splats

# Evolution parameters
DEFAULT_WORK_RESOLUTION = (128, 128)  # H, W
WORK_MAX_SIDE = 512
N_SPLATS = 512
POP_SIZE = 8
GENERATIONS = 7000
TOUR_K = 2  # Tournament selection size
ELITE_K = 4  # Number of elites to preserve
CXPB = 0.05  # Crossover probability

# Mutation parameters
MUTPB = 0.01  # Mutation probability
PROTECT_BEST_ELITE = True

# Rendering parameters
K_SIGMA = 3.0  # Gaussian extent multiplier
DEFAULT_TILE_SIZE = 32

# Genome constraints
MIN_SCALE_SPLATS = 0.4
MAX_SCALE_SPLATS = 0.15

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
    "rgb": 5.0, 
    "alpha": 5.0
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
REF_IMG = "reference.png"

# Video saving
SAVE_VIDEO = True
VIDEO_LEN = 10
FPS = 30
FRAME_EVERY = max(1, GENERATIONS // (FPS * VIDEO_LEN))

# Loss curve outputs
SAVE_LOSS_CURVE = True
LOSS_LOG_Y = False
