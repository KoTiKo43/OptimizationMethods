from typing import Callable
import numpy as np
from Common.datatypes import SearchResult

#PHI = 1.61803398874989484820
PSI = 0.61803398874989484820

def golden_ratio_multi(function: Callable[[np.ndarray], float], lhs: np.ndarray, rhs: np.ndarray, eps: float = 1e-6,
                 num_iterations: int = 1000) -> SearchResult:
    if lhs.shape != rhs.shape:
        return SearchResult.error("shape mismatch")
    
    cur_lhs = lhs.copy()
    cur_rhs = rhs.copy()

    xr = cur_lhs + PSI * (cur_rhs - cur_lhs)
    xl = cur_rhs - PSI * (cur_rhs - cur_lhs)
    fl = function(xl)
    fr = function (xr)

    index = 0
    while index < num_iterations and np.linalg.norm(cur_rhs - cur_lhs) >= 2*eps:
        if fl > fr:
            cur_lhs = xl
            xl = xr
            fl = fr
            xr = cur_lhs + PSI * (cur_rhs - cur_lhs)
            fr = function(xr)
        else:
            cur_rhs = xr
            xr = xl
            fr = fl
            xl = cur_rhs - PSI * (cur_rhs - cur_lhs)
            fl = function(xl)
        index += 1

    return SearchResult(
        method_name="golden_ratio multi",
        iterations=index,
        function_probes=index + 2,
        accuracy=np.linalg.norm(cur_rhs - cur_lhs) * 0.5,
        result=(cur_lhs + cur_rhs) * 0.5
    )
