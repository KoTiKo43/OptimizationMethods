from typing import Callable
from Common.datatypes import SearchResult

def fibonacci(function: Callable[[SearchResult], float], lhs: float, rhs: float, eps: float = 1e-6) -> SearchResult:
    lhs, rhs = (rhs, lhs) if lhs > rhs else (lhs, rhs)

    # fib_n0, fib_n1, fib_n2 = 1.0, 1.0, 2.0

    f_curr, f_next = 1.0, 1.0

    iters = 0
    value = (rhs - lhs) / eps
    while f_curr < value:
        iters += 1
        f_curr, f_next = f_next, f_curr + f_next
        # fib_n0, fib_n1 = fib_n1, fib_n2
        # fib_n2 += fib_n0

    xr = lhs + (f_curr / f_next) * (rhs - lhs)
    xl = lhs + ((f_next - f_curr) / f_next) * (rhs - lhs)
    fl = function(xl)
    fr = function(xr)

    for _ in range(iters):
        f_curr, f_next = f_next - f_curr, f_curr
        # fib_n0 = fib_n2 - fib_n1
        # fib_n1, fib_n2 = fib_n0, fib_n1
        if fl > fr:
            lhs = xl
            xl = xr
            fl = fr
            xr = lhs + (f_curr / f_next) * (rhs - lhs)
            # delta = abs(xr - xl) / 100
            # if abs(xr - xl) < delta:
            #     xr += delta
            fr = function(xr)
        else:
            rhs = xr
            xr = xl
            fr = fl
            xl = lhs + (f_next - f_curr) / f_next * (rhs - lhs)
            # delta = abs(xr - xl) / 100
            # if abs(xr - xl) < delta:
            #     xl -= delta
            fl = function(xl)
#    print(f'fibonacci: Accuracy {rhs - lhs}')
#    print(f'fibonacci: Function probe {iters + 2}')
    return SearchResult(
        method_name="fibonacci",
        iterations=iters,
        function_probes=iters + 2,
        accuracy=abs(rhs - lhs)  * 0.5,
        result=(lhs + rhs) * 0.5
    )
