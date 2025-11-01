from typing import Callable
import numpy as np
from Common.datatypes import SearchResult
from Common.functions import compute_gradient
from OneDimensionalMethods import fibonacci

def conjugate_gradient(function: Callable[[np.ndarray], float], x0: np.ndarray, max_iters: int = 1000, eps: float = 1e-6) -> SearchResult:
        
    x_cur = x0.copy()
    iters = 0
    
    grad_cur = compute_gradient(function, x_cur)
    func_probes = 2 * len(x_cur)
    
    s_cur = -grad_cur
    
    def f_along_gradient(lambda_val):
            return function(x_cur + lambda_val * s_cur)
    
    lambda_res = fibonacci(f_along_gradient, 0, 1)
    lambda_val = lambda_res.result
    func_probes += lambda_res.function_probes
    
    x_prev = x_cur.copy()
    grad_prev = grad_cur.copy()
    s_prev = s_cur.copy()
    
    x_cur = x_cur + lambda_val * s_prev
    
    for _ in range(max_iters):
        iters += 1
        
        grad_cur = compute_gradient(function, x_cur)
        func_probes += 2 * len(x_cur)
        
        omega = np.dot(grad_cur, grad_cur) / np.dot(grad_prev, grad_prev)
        
        s_cur = -grad_cur + omega * s_prev
        
        lambda_res = fibonacci(f_along_gradient, 0, 1)
        lambda_val = lambda_res.result
        func_probes += lambda_res.function_probes
        
        x_prev = x_cur.copy()
        grad_prev = grad_cur.copy()
        s_prev = s_cur.copy()
        
        x_cur = x_cur + lambda_val * s_cur
        
        if np.linalg.norm(x_cur - x_prev) < 2 * eps:
            break
        
    return SearchResult(
        method_name="conjugate_gradient",
        iterations=iters,
        function_probes=func_probes,
        accuracy=np.linalg.norm(x_cur - x_prev),
        result=x_cur
    )
    