from typing import Callable
import numpy as np
from Common.datatypes import SearchResult
from Common.functions import compute_hessian, compute_gradient

def method_newton_raphson(function: Callable[[float], float], x0: np.ndarray, max_iters: int = 1000, eps: float = 1e-6) -> SearchResult:
    x_cur = x0.copy()
    iters = 0
    func_probes = 0
    
    for _ in range(max_iters):
        iters += 1
        
        x_prev = x_cur.copy()
        
        x_cur = x_prev - np.dot(np.linalg.inv(compute_hessian(function, x_prev, func_probes)), compute_gradient(function, x_prev))
        func_probes += 2 * x_cur.size
        
        if np.linalg.norm(x_cur - x_prev) < 2 * eps:
            break
        
    return SearchResult(
        method_name="newton_raphson",
        iterations=iters,
        function_probes=func_probes,
        accuracy=np.linalg.norm(x_cur - x_prev),
        result=x_cur
    )
