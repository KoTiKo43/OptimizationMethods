from typing import Callable, List
import numpy as np
from Common.datatypes import SearchResult
from MultiDimensionalMethods import method_newton_raphson, conjugate_gradient
from functools import wraps

def penalty_internal_decorator(constraints: Callable[[float], float], lambda_val: float = 1.0):
    def decorator(func):
        @wraps(func)
        def wrapper(x: np.ndarray, *args, **kwargs):
            f_value = func(x, *args, **kwargs)

            penalty_values = []
            for constraint in constraints:
                constraint_value = constraint(x)
                if constraint_value > 0:
                    penalty_values.append(np.inf)
                else:
                    penalty_values.append(1.0 / (-constraint_value))

            return f_value + lambda_val * max(penalty_values)
        return wrapper
    return decorator

def penalty_external_decorator(inequality_constraints: List[Callable[[np.ndarray], float]], 
                     equality_constraints: List[Callable[[np.ndarray], float]], lambda_val: float = 1.0):
    def decorator(func):
        @wraps(func)
        def wrapper(x: np.ndarray, *args, **kwargs):
            f_value = func(x, *args, **kwargs)
            
            penalty = 0.0
            for constraint in inequality_constraints:
                constraint_val = constraint(x)
                
                if constraint_val > 0:
                    penalty += constraint_val**2
            
            for constraint in equality_constraints:
                penalty += constraint(x) ** 2

            return f_value + lambda_val * penalty
        return wrapper
    return decorator


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
        
        penalized_function = penalty_internal_decorator(constraints=constraints, 
                                                        lambda_val=lambda_val)(function)
        
        result = conjugate_gradient(penalized_function, x_cur, max_iters=max_iters, eps=eps)
        
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
        
        penalized_function = penalty_external_decorator(inequality_constraints=inequality_constraints, 
                                                        equality_constraints=equality_constraints, 
                                                        lambda_val=lambda_val)(function)
        
        result = method_newton_raphson(penalized_function, x_cur, max_iters=max_iters, eps=eps)
        
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
# Декоратор, аргументы - список функций которые могут быть штрафными. Два декоратора - internal penalty, external penalty
# Функцию, которую хочешь покрыть штрафами, декорируешь и передаёшь в базовый алгоритм