from typing import Callable
from Common.datatypes import SearchResult

#PHI = 1.61803398874989484820
PSI = 0.61803398874989484820

def golden_ratio(function: Callable[[SearchResult], float], lhs: float, rhs: float, eps: float = 1e-6,
                 num_iterations: int = 1000) -> SearchResult:
    
    lhs, rhs = (rhs, lhs) if lhs > rhs else (lhs, rhs)

    xr = lhs + PSI * (rhs - lhs)
    xl = rhs - PSI * (rhs - lhs)
    fl = function(xl)
    fr = function (xr)

    index = 0
    while index < num_iterations and abs(rhs - lhs) >= 2 * eps:
        if fl > fr:
            lhs = xl
            xl = xr
            fl = fr
            xr = lhs + PSI * (rhs - lhs)
            fr = function(xr)
        else:
            rhs = xr
            xr = xl
            fr = fl
            xl = rhs - PSI * (rhs - lhs)
            fl = function(xl)
        index += 1
#    print(f'golden_ratio: Accuracy {rhs - lhs}')
#    print(f'golden_ratio: Function probe {index + 2}')
    return SearchResult(
        method_name="golden_ratio",
        iterations=index,
        function_probes=index + 2,
        accuracy=abs(rhs - lhs) * 0.5,
        result=(lhs + rhs) * 0.5
    )
