from typing import Callable, List
import numpy as np
from Common.datatypes import SearchResult
from MultiDimensionalMethods import method_newton_raphson, conjugate_gradient

def penalty_internal(function: Callable[[float], float], constraints: List[Callable[[np.ndarray], float]], x0: np.ndarray, 
                     initial_lambda: float = 1.0, lambda_multiplier: float = 0.1, min_lambda: float = 1e-6, eps: float = 1e-6, 
                     max_iters: int = 1000, max_lambda_steps: int = 20) -> SearchResult:
    for constraint in constraints:
        if constraint(x0) > 0:
            # Ошибка: начальная точка не удовлетворяет ограничениям
            return SearchResult.error(error_message="error, starting point invalid", method_name="penalty_internal")
    
    x_cur = x0.copy()
    lambda_val = initial_lambda
    
    total_iterations = 0
    total_function_probes = 0
    final_accuracy = eps
    
    for _ in range(max_lambda_steps):
        if lambda_val < min_lambda:
            break
        
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
        
        result = conjugate_gradient(augmented_function, x_cur, max_iters=max_iters, eps=eps)
        
        total_iterations += result.iterations
        total_function_probes += result.function_probes
        final_accuracy = result.accuracy
        
        x_cur = result.result
        
        lambda_val *= lambda_multiplier
    
    return SearchResult(
        method_name="penalty_internal",
        iterations=total_iterations,
        function_probes=total_function_probes,
        accuracy=final_accuracy,
        result=x_cur
    )
    

def penalty_external(function: Callable[[float], float], inequality_constraints: List[Callable[[np.ndarray], float]], 
                     equality_constraints: List[Callable[[np.ndarray], float]], x0: np.ndarray, 
                     initial_lambda: float = 0.1, lambda_multiplier: float = 10.0, min_lambda: float = 1e-6, eps: float = 1e-6, 
                     max_iters: int = 1000, max_lambda_steps: int = 20) -> SearchResult:
    
    x_cur = x0.copy()
    lambda_val = initial_lambda
    
    total_iterations = 0
    total_function_probes = 0
    final_accuracy = eps
    
    for _ in range(max_lambda_steps):
        if lambda_val < min_lambda:
            break
        
        def penalty_function(x):
            penalty = 0.0
            for constraint in inequality_constraints:
                constraint_val = constraint(x)
                
                if constraint_val > 0:
                    penalty += constraint_val**2
            
            for constraint in equality_constraints:
                penalty += constraint(x) ** 2
                
            return lambda_val * penalty
            
        def augmented_function(x):
            return function(x) + penalty_function(x)
        
        result = method_newton_raphson(augmented_function, x_cur, max_iters=max_iters, eps=eps)
        
        total_iterations += result.iterations
        total_function_probes += result.function_probes
        final_accuracy = result.accuracy
        
        x_cur = result.result
        
        lambda_val *= lambda_multiplier
    
    return SearchResult(
        method_name="penalty_external",
        iterations=total_iterations,
        function_probes=total_function_probes,
        accuracy=final_accuracy,
        result=x_cur
    )
