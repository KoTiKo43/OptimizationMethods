import numpy as np
from Common.datatypes import SearchResult
import OneDimensionalMethods as odm
import MultiDimensionalMethods as mdm

#TODO pip install pydantic

def test_func(x: float) -> float:
    return (x - 2)**2 + 1

def quadratic_sphere(x: np.ndarray) -> float:
    center = np.array([2.0, 3.0, 1.0])
    return np.sum((x - center)**2)

def quadratic_sum(x: np.ndarray) -> float:
    return x[0]**2 + x[1]**2

def constraint1(x: np.ndarray) -> float:
    return x[0] + x[1] - 1

def constraint2(x: np.ndarray) -> float:
    return -x[0]

def constraint3(x: np.ndarray) -> float:
    return -x[1]

def methods_one_dim():
    lhs, rhs = 0.0, 5.0
    # print('\n'.join(map(lambda val:str(isinstance(val, float)), (lhs, rhs))))
    eps = 1e-6
    result_bisection: SearchResult = odm.bisection(test_func, lhs, rhs, eps)
    result_golden_ratio: SearchResult = odm.golden_ratio(test_func, lhs, rhs, eps)
    result_fibonacci: SearchResult = odm.fibonacci(test_func, lhs, rhs, eps * 2.5)
    
    print("Results:")
    print(result_bisection)
    print(result_golden_ratio)
    print(result_fibonacci)
    print(f'Accuracy divercity between golden_ratio and fibonacci: {abs(result_golden_ratio.accuracy - result_fibonacci.accuracy)}')

def methods_multi_dim():
    lhs = np.array((0.0, 0.0, 0.0))
    rhs = np.array((5.0, 5.0, 5.0))
    eps = 1e-6
    result_bisection_multi = mdm.bisection_multi(quadratic_sphere, lhs, rhs, eps)
    result_golden_ratio_multi = mdm.golden_ratio_multi(quadratic_sphere, lhs, rhs, eps)
    result_fibonacci_multi = mdm.fibonacci_multi(quadratic_sphere, lhs, rhs, eps * 2.5)
    
    result_per_coord_descent = mdm.per_coord_descent(quadratic_sphere, lhs)
    result_gradient_descent = mdm.gradient_descent(quadratic_sphere, lhs)
    result_conjugate_gradient = mdm.conjugate_gradient(quadratic_sphere, lhs)
    result_method_newton_raphson = mdm.method_newton_raphson(quadratic_sphere, lhs)
    
    x0_internal = np.array((0.5, 0.2))
    x0_external = np.array((2.0, 2.0))
    result_penalty_internal = mdm.penalty_internal(quadratic_sum, 
                                                   constraints=[constraint1, constraint2, constraint3], x0=x0_internal)
    result_penalty_external = mdm.penalty_external(quadratic_sum, 
                                                   inequality_constraints=[constraint2, constraint3], 
                                                   equality_constraints=[constraint1], x0=x0_external)
    
    print("Results:")
    print(result_bisection_multi)
    print(result_golden_ratio_multi)
    print(result_fibonacci_multi)
    
    print(result_per_coord_descent)
    print(result_gradient_descent)
    print(result_conjugate_gradient)
    print(result_method_newton_raphson)
    
    print("Penalty functions:")
    print(result_penalty_internal)
    print(result_penalty_external)


if __name__ == "__main__":
    # methods_one_dim()
    methods_multi_dim()
