from collections import namedtuple

class SearchResult(namedtuple(
    'SearchResult', 
    ['method_name', 'iterations', 'function_probes', 'accuracy', 'result']
)):
    
    def __str__(self):
        return (f"{self.method_name}: "
                f"result = {self.result}, "
                f"iterations = {self.iterations}, "
                f"function_probes = {self.function_probes}, "
                f"accuracy = {self.accuracy}")
    
    @classmethod
    def error(cls):
        return cls(
            method_name="error",
            iterations=0,
            function_probes=0,
            accuracy=0,
            result=0
        )
