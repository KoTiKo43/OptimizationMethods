from typing import Callable
import numpy as np
from Common.datatypes import SearchResult

def fibonacci_multi(function: Callable[[SearchResult], float], lhs: np.ndarray, rhs: np.ndarray, eps: float = 1e-6) -> SearchResult:
    if lhs.shape != rhs.shape:
        return SearchResult(
        method_name="error",
        iterations=0,
        function_probes=0,
        accuracy=0,
        result=0
    )
        
    cur_lhs = lhs.copy()
    cur_rhs = rhs.copy()

    fib_n0, fib_n1, fib_n2 = 1.0, 1.0, 2.0

    iters = 0
    value = np.linalg.norm(cur_rhs - cur_lhs) / eps
    while fib_n2 < value:
        iters += 1
        fib_n0, fib_n1 = fib_n1, fib_n2
        fib_n2 += fib_n0

    xr = cur_lhs + (fib_n1 / fib_n2) * (cur_rhs - cur_lhs)
    xl = cur_lhs + (fib_n0 / fib_n2) * (cur_rhs - cur_lhs)
    fl = function(xl)
    fr = function(xr)

    for _ in range(iters):
        fib_n0 = fib_n2 - fib_n1
        fib_n1, fib_n2 = fib_n0, fib_n1
        if fl > fr:
            cur_lhs = xl
            xl = xr
            fl = fr
            xr = cur_lhs + (fib_n1 / fib_n2) * (cur_rhs - cur_lhs)
            delta = np.linalg.norm(xr - xl) / 100
            if np.linalg.norm(xr - xl) < delta:
                xr += (cur_rhs - cur_lhs) / np.linalg.norm(cur_rhs - cur_lhs) * delta
            fr = function(xr)
        else:
            cur_rhs = xr
            xr = xl
            fr = fl
            xl = cur_lhs + ((fib_n2 - fib_n1) / fib_n2) * (cur_rhs - cur_lhs)
            delta = np.linalg.norm(xr - xl) / 100
            if np.linalg.norm(xr - xl) < delta:
                xl += (cur_rhs - cur_lhs) / np.linalg.norm(cur_rhs - cur_lhs) * delta
            fl = function(xl)
#    print(f'fibonacci: Accuracy {rhs - lhs}')
#    print(f'fibonacci: Function probe {iters + 2}')
    return SearchResult(
        method_name="fibonacci multi",
        iterations=iters,
        function_probes=iters + 2,
        accuracy=np.linalg.norm(cur_rhs - cur_lhs) / 2,
        result=(cur_lhs + cur_rhs) * 0.5
    )
