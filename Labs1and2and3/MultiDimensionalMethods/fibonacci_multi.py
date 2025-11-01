from typing import Callable
import numpy as np
from Common.datatypes import SearchResult

def fibonacci_multi(function: Callable[[np.ndarray], float], lhs: np.ndarray, rhs: np.ndarray, eps: float = 1e-6) -> SearchResult:
    if lhs.shape != rhs.shape:
        return SearchResult.error()
        
    cur_lhs = lhs.copy()
    cur_rhs = rhs.copy()

    f_cur, f_next = 1.0, 1.0

    iters = 0
    value = np.linalg.norm(cur_rhs - cur_lhs) / eps
    while f_next < value:
        iters += 1
        f_cur, f_next = f_next, f_cur + f_next

    xr = cur_lhs + (f_cur / f_next) * (cur_rhs - cur_lhs)
    xl = cur_lhs + ((f_next - f_cur) / f_next) * (cur_rhs - cur_lhs)
    fl = function(xl)
    fr = function(xr)

    for _ in range(iters):
        f_cur, f_next = f_next - f_cur, f_cur

        if fl > fr:
            cur_lhs = xl
            xl = xr
            fl = fr
            xr = cur_lhs + (f_cur / f_next) * (cur_rhs - cur_lhs)
            # delta = np.linalg.norm(xr - xl) / 100
            # if np.linalg.norm(xr - xl) < delta:
            #     xr += (cur_rhs - cur_lhs) / np.linalg.norm(cur_rhs - cur_lhs) * delta
            fr = function(xr)
        else:
            cur_rhs = xr
            xr = xl
            fr = fl
            xl = cur_lhs + ((f_next - f_cur) / f_next) * (cur_rhs - cur_lhs)
            # delta = np.linalg.norm(xr - xl) / 100
            # if np.linalg.norm(xr - xl) < delta:
            #     xl += (cur_rhs - cur_lhs) / np.linalg.norm(cur_rhs - cur_lhs) * delta
            fl = function(xl)
#    print(f'fibonacci: Accuracy {rhs - lhs}')
#    print(f'fibonacci: Function probe {iters + 2}')
    return SearchResult(
        method_name="fibonacci multi",
        iterations=iters,
        function_probes=iters + 2,
        accuracy=np.linalg.norm(cur_rhs - cur_lhs) * 0.5,
        result=(cur_lhs + cur_rhs) * 0.5
    )
