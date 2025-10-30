from typing import Callable, List
import numpy as np
from Common.datatypes import SearchResult
from MultiDimensionalMethods import method_newton_raphson

def penalty_internal(function: Callable[[float], float], constraints: List[Callable[[np.ndarray], float]], x0: np.ndarray, 
                     lambda_val: float = 1.0, eps: float = 1e-6, max_iters: int = 1000) -> SearchResult:
    for constraint in constraints:
        if constraint(x0) > 0:
            # Ошибка: начальная точка не удовлетворяет ограничениям
            return SearchResult(
                method_name='internal_penalty',
                iterations=-1,
                function_probes=-1,
                accuracy=-1,
                result=-1
            )
            
    def penalty_function(x):
        penalty = 0.0
        for constraint in constraints:
            constraint_value = constraint(x)
            
            if constraint_value >= 0:
                penalty = 1e10 # Большой штраф за несоответствие условию
                
            penalty += 1.0 / (-constraint_value)
            
        return lambda_val * penalty
    
    def augmented_function(x):
        return function(x) + penalty_function(x)
            
    return method_newton_raphson(augmented_function, x0, max_iters=max_iters, eps=eps)
    

def penalty_external(function: Callable[[float], float], inequality_constraints: List[Callable[[np.ndarray], float]], 
                     equality_constraints: List[Callable[[np.ndarray], float]], x0: np.ndarray, 
                     lambda_val: float = 1.0, eps: float = 1e-6, max_iters: int = 1000) -> SearchResult:
    def penalty_function(x):
        penalty = 0.0
        for constraint in inequality_constraints:
            constraint_val = constraint(x)
            
            if constraint_val > 0:
                penalty += constraint_val**2
        
        for constraint in equality_constraints:
            penalty += constraint(x)
            
        return lambda_val * penalty
        
    def augmented_function(x):
        return function(x) + penalty_function(x)
    
    return method_newton_raphson(augmented_function, x0, max_iters=max_iters, eps=eps)
