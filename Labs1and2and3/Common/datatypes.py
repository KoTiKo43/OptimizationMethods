import numpy as np
from collections import namedtuple
from dataclasses import dataclass
from typing import List, Dict
from enum import Enum

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
    def error(cls, error_message="error", method_name="error"):
        return cls(
            method_name=method_name,
            iterations=0,
            function_probes=0,
            accuracy=0,
            result=error_message
        )

class ProblemType(Enum):
    MAX = "max"
    MIN = "min"

class CompareSign(Enum):
    LESS_EQUAL = "<="
    EQUAL = "="
    GREATER_EQUAL = ">="

@dataclass
class SimplexResult:
    solution: np.ndarray                    # Вектор решения (основные переменные)
    target_value: float                     # Значение целевой функции в оптимуме
    iterations: int                         # Количество итераций
    is_feasible: bool                       # Есть ли допустимое решение
    is_optimal: bool                        # Достигнут ли оптимум
    is_unbounded: bool                      # Задача неограничена?
    is_infeasible: bool                     # Задача несовместна?
    problem_type: ProblemType               # Тип задачи (MAX/MIN)
    basic_vars: List[int]                   # Индексы базисных переменных в конце
    table_history: List[np.ndarray] = None  # История симплекс-таблиц (опционально)
    
    def __str__(self) -> str:
        lines = []
        lines.append("=" * 70)
        lines.append("SIMPLEX METHOD RESULT")
        lines.append("=" * 70)
        
        if not self.is_feasible:
            if self.is_unbounded:
                lines.append("PROBLEM IS UNBOUNDED")
                lines.append("The objective function can be improved indefinitely.")
            elif self.is_infeasible:
                lines.append("NO FEASIBLE SOLUTION")
                lines.append("The constraints cannot be satisfied simultaneously.")
            else:
                lines.append("NO SOLUTION FOUND")
        else:
            lines.append(f"{'MAXIMUM' if self.problem_type == ProblemType.MAX else 'MINIMUM'} FOUND")
            lines.append(f"Iterations: {self.iterations}")
            lines.append(f"Optimal value: F = {self.target_value:.6f}")
            
            lines.append("\nSolution vector:")
            for i, val in enumerate(self.solution):
                lines.append(f"  x{i+1} = {val:.6f}")
        
        lines.append("=" * 70)
        return "\n".join(lines)
    
    def get_solution_dict(self) -> Dict[str, float]:
        return {f'x{i+1}': np.round(val, 4).item() for i, val in enumerate(self.solution)}
