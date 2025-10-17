from typing import Callable
import numpy as np
from Common.datatypes import SearchResult
from OneDimensionalMethods import fibonacci

def per_coord_descent(function: Callable[[float], float], x0: np.ndarray, max_iters: int = 1000, step: float = 0.1, 
                      eps: float = 1e-6) -> SearchResult:
    n = len(x0) # Размерность
    x_cur = x0.copy()
    func_probes = 0
    iters = 0
    
    for _ in range(max_iters):
        x_prev = x_cur.copy()
        
        for j in range(n):
            e_j = np.zeros(n) # Единичный вектор для текущей координаты
            e_j[j] = 1.0
            
            xl = x_cur - eps * e_j
            xr = x_cur + eps * e_j
            
            fl = function(xl)
            fr = function(xr)
            func_probes += 2
            
            if fl > fr:
                # Преобразуем промежуток неопределённости
                search_low = x_cur[j]
                search_high = x_cur[j] + step
            else:
                # Преобразуем промежуток неопределённости
                search_low = x_cur[j] - step
                search_high = x_cur[j]
                
            # Функция для одномерного поиска вдоль j-й координаты
            def func_1d(alpha):
                x_temp = x_cur.copy()
                x_temp[j] = alpha
                return function(x_temp)
            
            fib_result = fibonacci(func_1d, search_low, search_high, eps)
            func_probes += fib_result.function_probes
            
            x_cur[j] = fib_result.result
                
        iters += 1
        
        if np.linalg.norm(x_cur - x_prev) < eps:
            break
        
    return SearchResult(
        method_name = "per_coordinate_descent",
        iterations = iters,
        function_probes = func_probes,
        accuracy = np.linalg.norm(x_cur - x_prev),
        result = x_cur
    )
