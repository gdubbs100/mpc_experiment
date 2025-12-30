from typing import Callable

def numerical_grad(
        landscape_func: Callable[[float], float], 
        x: float, 
        eps:float=1e-5
    ) -> float:
    return (landscape_func(x + eps) - landscape_func(x - eps)) / (2 * eps)