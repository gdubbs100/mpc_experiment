import torch
import numpy as np
from typing import Callable

def numerical_grad(
        landscape_func: Callable[[float], float], 
        x: float, 
        eps:float=1e-5
    ) -> float:
    return (landscape_func(x + eps) - landscape_func(x - eps)) / (2 * eps)

def sign(u: torch.Tensor):
    if isinstance(u, torch.Tensor):
        return torch.sign(u)
    else:
        return np.sign(u)

def sqrt(s: torch.Tensor):
    if isinstance(s, torch.Tensor):
        return torch.sqrt(s)
    else:
        return np.sqrt(s)