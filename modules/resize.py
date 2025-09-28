import numpy as np
import torch
from typing import Tuple


def choose_work_size(Ht:int, Wt:int, max_side:int=128) -> Tuple[int,int]:
    """Return (H_work, W_work) preserving aspect ratio with longest side = max_side."""
    if Ht >= Wt:
        Hf = max_side
        Wf = max(1, int(round(Wt * Hf / Ht)))
    else:
        Wf = max_side
        Hf = max(1, int(round(Ht * Wf / Wt)))
    return Hf, Wf


def scale_genome_pixels_anisotropic(ind: torch.Tensor, sH: float, sW: float) -> torch.Tensor:
    """
    Anisotropic scaling for a genome that uses (a_log, b_log, theta).
    - sigma_x scales with sW  -> a_log += log(sW)
    - sigma_y scales with sH  -> b_log += log(sH)
    - theta unchanged
    """
    out = ind.clone()
    out[:, 2] += float(np.log(sW))   # a_log (horizontal radius)
    out[:, 3] += float(np.log(sH))   # b_log (vertical radius)
    # theta at [:,4] unchanged
    return out
