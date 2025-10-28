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