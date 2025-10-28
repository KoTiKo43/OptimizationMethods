from typing import Callable
import numpy as np

def compute_gradient(f: Callable[[float], float], x: np.ndarray, dx: float = 1e-6) -> np.ndarray:
        grad = np.zeros_like(x)
        
        for i in range(len(x)):
            x_plus = x.copy()
            x_minus = x.copy()
            
            x_plus[i] += dx
            x_minus[i] -= dx
            
            grad[i] = (f(x_plus) - f(x_minus)) / (2 * dx)
            
        return grad   
