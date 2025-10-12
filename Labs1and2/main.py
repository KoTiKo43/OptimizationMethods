import OneDimensionalMethods
from collections import namedtuple

SearchResult = namedtuple(
    'SearchResult', 
    ['method_name', 'iterations', 'function_probes', 'accuracy', 'result']
)

def search_result_str(self):
    return (f"{self.method_name}: "
            f"result = {self.result}, "
            f"iterations = {self.iterations}, "
            f"function_probes = {self.function_probes}, "
            f"accuracy = {self.accuracy}")

SearchResult.__str__ = search_result_str


def test_func(x: float) -> float:
    return (x - 2)**2 + 1

def methods_one_dim():
    result_bisection = OneDimensionalMethods.bisection(test_func, 0, 5)
    result_golden_ratio = OneDimensionalMethods.golden_ratio(test_func, 0, 5)
    result_fibonacci = OneDimensionalMethods.fibonacci(test_func, 0, 5)
    
    print("Results:")
    print(result_bisection)
    print(result_golden_ratio)
    print(result_fibonacci)

def methods_multi_dim():
    pass


if __name__ == "__main__":
    methods_one_dim()