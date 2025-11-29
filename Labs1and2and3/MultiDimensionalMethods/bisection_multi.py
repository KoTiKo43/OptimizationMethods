from typing import Callable
import numpy as np
from Common.datatypes import SearchResult

def bisection_multi(function: Callable[[np.ndarray], float], lhs: np.ndarray, rhs: np.ndarray, eps: float = 1e-6, 
              num_iterations: int = 1000) -> SearchResult:
    
    if lhs.shape != rhs.shape:
        return SearchResult.error("shape mismatch")
    
    cur_lhs = lhs.copy()
    cur_rhs = rhs.copy()

    dir_vec = (cur_rhs - cur_lhs) / np.linalg.norm(cur_rhs - cur_lhs) * 0.1 * eps

    index = 0
    while index < num_iterations and np.linalg.norm(cur_rhs - cur_lhs) >= 2 * eps:
        xc = (cur_lhs + cur_rhs) * 0.5
        xl = xc - dir_vec
        xr = xc + dir_vec
        if function(xl) > function(xr):
            cur_lhs = xl
        else:
            cur_rhs = xr
        index += 1

#    print(f'bisection: Accuracy {rhs - lhs}')
#    print(f'bisection: Function probe {index * 2}')

    return SearchResult(
        method_name="bisection multi",
        iterations=index,
        function_probes=index * 2,
        accuracy=np.linalg.norm(cur_rhs - cur_lhs) * 0.5,
        result=(cur_lhs + cur_rhs) * 0.5
    )
