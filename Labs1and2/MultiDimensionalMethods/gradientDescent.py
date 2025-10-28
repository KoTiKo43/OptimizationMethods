from typing import Callable
import numpy as np
from Common.datatypes import SearchResult
from OneDimensionalMethods import fibonacci

def gradient_descent(function: Callable[[float], float], x0: np.ndarray, step: float = 1.0, max_iters = 1000, eps: float = 1e-6) -> SearchResult:
    
    def compute_gradient(f: Callable[[float], float], x: np.ndarray, dx: float = 1e-6) -> np.ndarray:
        grad = np.zeros_like(x)
        
        for i in range(n):
            x_plus = x.copy()
            x_minus = x.copy()
            
            x_plus[i] += dx
            x_minus[i] -= dx
            
            grad[i] = (f(x_plus) - f(x_minus)) / (2 * dx)
            
        return grad   
    
    n = len(x0) # Размерность
    x_cur = x0.copy()
    func_probes = 0
    iters = 0
    
    for _ in range(max_iters):
        grad = compute_gradient(function, x_cur)
        func_probes += 2 * len(x_cur)
        
        def f_along_gradient(lambda_val):
            return function(x_cur - lambda_val * grad)
        
        lambda_res = fibonacci(f_along_gradient, 0, 1)
        
        step = lambda_res.result
        func_probes += lambda_res.function_probes
        
        x_prev = x_cur.copy()
        
        x_cur = x_cur - step * grad
        
        if np.linalg.norm(x_cur - x_prev) < 2 * eps:
            break
        
        iters += 1
        
    return SearchResult(
        method_name="gradient_descent",
        iterations=iters,
        function_probes=func_probes,
        accuracy=np.linalg.norm(x_cur - x_prev),
        result=x_cur
    )
