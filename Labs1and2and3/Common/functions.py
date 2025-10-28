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
    
def compute_hessian(f: Callable[[float], float], x: np.ndarray, func_probes: int, dx: float = 1e-6) -> np.ndarray:
    n = len(x)
    hessian = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i, n):
            if i == j:
                x_plus = x.copy()
                x_minus = x.copy()
                x_plus[i] += dx
                x_minus[i] -= dx
                hessian[i, i] = (f(x_plus) - 2 * f(x) + f(x_minus)) / dx**2
                
                func_probes += 3
            else:
                x_pp = x.copy(); x_pp[i] += dx; x_pp[j] += dx
                x_pm = x.copy(); x_pm[i] += dx; x_pm[j] -= dx
                x_mp = x.copy(); x_mp[i] -= dx; x_mp[j] += dx
                x_mm = x.copy(); x_mm[i] -= dx; x_mm[j] -= dx
                
                hessian[i, j] = (f(x_pp) - f(x_pm) + f(x_mm) - f(x_mp)) / (4 * dx**2)
                hessian[j, i] = hessian[i, j] # Пользуемся симметрией матрицы Гессе
                
                func_probes += 4
                
    return hessian
