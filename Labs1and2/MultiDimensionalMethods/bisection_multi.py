from typing import Callable
import numpy as np
from Common.datatypes import SearchResult

def bisection_multi(function: Callable[[float], float], lhs: np.ndarray, rhs: np.ndarray, eps: float = 1e-6, 
              num_iterations: int = 1000) -> SearchResult:
    
    if lhs.shape != rhs.shape:
        return SearchResult(
        method_name="error",
        iterations=0,
        function_probes=0,
        accuracy=0,
        result=0
    )

    dir_vec = np.linalg.norm(rhs - lhs) * 0.1 * eps
    
    cur_lhs = lhs.copy()
    cur_rhs = rhs.copy()

    index = 0
    while index < num_iterations and np.linalg.norm(cur_rhs - cur_lhs) >= 2*eps:
        xc = (cur_lhs + cur_rhs) * 0.5
        if function(xc - eps) > function(xc + eps):
            cur_lhs = xc - dir_vec
        else:
            cur_rhs = xc + dir_vec
        index += 1

#    print(f'bisection: Accuracy {rhs - lhs}')
#    print(f'bisection: Function probe {index * 2}')

    return SearchResult(
        method_name="bisection multi",
        iterations=index,
        function_probes=index * 2,
        accuracy=np.linalg.norm(cur_rhs - cur_lhs) / 2,
        result=(cur_lhs + cur_rhs) * 0.5
    )

