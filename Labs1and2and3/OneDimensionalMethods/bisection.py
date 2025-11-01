from typing import Callable
from Common.datatypes import SearchResult

def bisection(function: Callable[[SearchResult], float], lhs: float, rhs: float, eps: float = 1e-6, 
              num_iterations: int = 1000) -> SearchResult:
    
    lhs, rhs = (rhs, lhs) if lhs > rhs else (lhs, rhs)

    index = 0
    while index < num_iterations and abs(rhs - lhs) >= 2*eps:
        xc = (lhs + rhs) * 0.5
        if function(xc - eps) > function(xc + eps):
            lhs = xc - eps*0.1
        else:
            rhs = xc + eps*0.1
        index += 1

#    print(f'bisection: Accuracy {rhs - lhs}')
#    print(f'bisection: Function probe {index * 2}')

    return SearchResult(
        method_name="bisection",
        iterations=index,
        function_probes=index * 2,
        accuracy=abs(rhs - lhs) / 2,
        result=(lhs + rhs) * 0.5
    )
