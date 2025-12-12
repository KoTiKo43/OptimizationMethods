from collections import namedtuple

class SearchResult(namedtuple(
    'SearchResult', 
    ['method_name', 'iterations', 'function_probes', 'accuracy', 'result']
)):
    
    def __str__(self):
        return (f"{self.method_name}:\n"
                f"result          = {self.result},\n"
                f"iterations      = {self.iterations},\n"
                f"function_probes = {self.function_probes},\n"
                f"accuracy        = {self.accuracy}\n\n")
    
    @classmethod
    def error(cls):
        return cls(
            method_name="error",
            iterations=0,
            function_probes=0,
            accuracy=0,
            result=0
        )
