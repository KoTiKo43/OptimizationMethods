import numpy as np
from Common.datatypes import SearchResult
import OneDimensionalMethods
import MultiDimensionalMethods

def test_func(x: float) -> float:
    return (x - 2)**2 + 1

def quadratic_sphere(x: np.ndarray) -> float:
    center = np.array([2.0, 3.0, 1.0])
    return np.sum((x - center)**2)

def methods_one_dim():
    lhs, rhs = 0, 5
    result_bisection = OneDimensionalMethods.bisection(test_func, lhs, rhs)
    result_golden_ratio = OneDimensionalMethods.golden_ratio(test_func, lhs, rhs)
    result_fibonacci = OneDimensionalMethods.fibonacci(test_func, lhs, rhs)
    
    print("Results:\n")
    print(result_bisection)
    print(result_golden_ratio)
    print(result_fibonacci)

def methods_multi_dim():
    lhs = np.array((0.0, 0.0, 0.0))
    rhs = np.array((5.0, 5.0, 5.0))
    result_bisection_multi = MultiDimensionalMethods.bisection_multi(quadratic_sphere, lhs, rhs)
    result_golden_ratio_multi = MultiDimensionalMethods.golden_ratio_multi(quadratic_sphere, lhs, rhs)
    result_fibonacci_multi = MultiDimensionalMethods.fibonacci_multi(quadratic_sphere, lhs, rhs)
    result_per_coord_descent = MultiDimensionalMethods.per_coord_descent(quadratic_sphere, lhs)
    
    print("Results:")
    print(result_bisection_multi)
    print(result_golden_ratio_multi)
    print(result_fibonacci_multi)
    print(result_per_coord_descent)


if __name__ == "__main__":
    # methods_one_dim()
    methods_multi_dim()
