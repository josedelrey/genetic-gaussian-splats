import numpy as np
import torch
from typing import Tuple


def choose_work_size(Ht:int, Wt:int, max_side:int=128) -> Tuple[int,int]:
    if Ht >= Wt:
        Hf = max_side
        Wf = max(1, int(round(Wt * Hf / Ht)))
    else:
        Wf = max_side
        Hf = max(1, int(round(Ht * Wf / Wt)))
    return Hf, Wf


def scale_genome_pixels_anisotropic(ind: torch.Tensor, sH: float, sW: float) -> torch.Tensor:
    out = ind.clone()
    out[:, 2] += float(np.log(sW))
    out[:, 3] += float(np.log(sH))
    return out
