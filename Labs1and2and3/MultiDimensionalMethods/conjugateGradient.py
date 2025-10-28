from typing import Callable
import numpy as np
from Common.datatypes import SearchResult
from Common.computeGradient import compute_gradient
from OneDimensionalMethods import fibonacci

def conjucate_gradient(function: Callable[[float], float], x0: np.ndarray, max_iters: int = 1000, eps: float = 1e-6) -> SearchResult:
    x_cur = x0.copy()
    
    s0 = -compute_gradient(function(x_cur))
    
    