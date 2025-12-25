import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
from Common.datatypes import ProblemType, CompareSign, SimplexResult


def simplex_method(
    A: np.ndarray,                  # Матрица коэффициентов ограничений (m x n)
    b: np.ndarray,                  # Вектор правых частей ограничений (m)
    c: np.ndarray,                  # Вектор коэффициентов целевой функции (n)
    signs: List[CompareSign],       # Список знаков сравнения для ограничений
    problem_type: ProblemType,      # Тип задачи (MAX или MIN)
    record_history: bool = False,   # Флаг записи истории симплекс-таблиц
    max_iterations: int = 1000
) -> SimplexResult:
    m, n = A.shape  # m ограничений, n переменных
    
    # 1. Преобразование задачи к канонической форме
    # Все ограничения должны быть в виде равенств с неотрицательными переменными
    
    # Определяем количество дополнительных переменных
    n_slack = 0  # дополнительные переменные для неравенств
    n_surplus = 0  # избыточные переменные для неравенств >=
    n_artificial = 0  # искусственные переменные
    
    for sign in signs:
        if sign == CompareSign.LESS_EQUAL:
            n_slack += 1
        elif sign == CompareSign.GREATER_EQUAL:
            n_slack += 1  # для >= тоже нужна остаточная переменная, но с отрицательным коэффициентом
            n_surplus += 1
            n_artificial += 1  # для двухфазного метода
        elif sign == CompareSign.EQUAL:
            n_artificial += 1
    
    # Общее количество переменных в расширенной задаче
    total_vars = n + n_slack + n_surplus + n_artificial
    
    # Создаем расширенную матрицу A
    A_ext = np.zeros((m, total_vars))
    
    # Копируем исходные коэффициенты
    A_ext[:, :n] = A.copy()
    
    # Индексы переменных
    slack_idx = n
    surplus_idx = n + n_slack
    artificial_idx = n + n_slack + n_surplus
    
    # Списки для отслеживания типов переменных
    slack_vars = []
    surplus_vars = []
    artificial_vars = []
    
    # Заполняем дополнительные переменные
    row = 0
    slack_count = 0
    surplus_count = 0
    artificial_count = 0
    
    for i, sign in enumerate(signs):
        if sign == CompareSign.LESS_EQUAL:
            # Добавляем slack переменную
            A_ext[i, slack_idx + slack_count] = 1
            slack_vars.append(slack_idx + slack_count)
            slack_count += 1
            
        elif sign == CompareSign.GREATER_EQUAL:
            # Добавляем surplus переменную (со знаком -)
            A_ext[i, surplus_idx + surplus_count] = -1
            surplus_vars.append(surplus_idx + surplus_count)
            surplus_count += 1
            
            # Добавляем искусственную переменную для двухфазного метода
            A_ext[i, artificial_idx + artificial_count] = 1
            artificial_vars.append(artificial_idx + artificial_count)
            artificial_count += 1
            
        elif sign == CompareSign.EQUAL:
            # Добавляем искусственную переменную
            A_ext[i, artificial_idx + artificial_count] = 1
            artificial_vars.append(artificial_idx + artificial_count)
            artificial_count += 1
    
    # 2. Двухфазный симплекс-метод (если есть искусственные переменные)
    if n_artificial > 0:
        # Фаза 1: Минимизация суммы искусственных переменных
        # Целевая функция фазы 1: -сумма искусственных переменных (или минимизация)
        c_phase1 = np.zeros(total_vars)
        for idx in artificial_vars:
            c_phase1[idx] = -1  # Минимизируем сумму искусственных переменных
        
        # Начальный базис: искусственные переменные
        basic_vars = artificial_vars.copy()
        
        # Создаем симплекс-таблицу для фазы 1
        table = np.zeros((m + 1, total_vars + 1))
        table[:m, :total_vars] = A_ext
        table[:m, -1] = b.copy()
        
        # Целевая функция фазы 1
        table[-1, :total_vars] = -c_phase1  # Берем с минусом для приведения к стандартной форме
        
        # Пересчитываем целевую строку для базисных переменных
        for i, var_idx in enumerate(basic_vars):
            table[-1] -= table[-1, var_idx] * table[i]
        
        # Решаем фазу 1
        result_phase1 = _simplex_iterations(
            table, basic_vars,
            max_iterations=max_iterations,
            record_history=record_history,
        )
        
        # Проверяем результат фазы 1
        optimal_value_phase1 = -table[-1, -1]  # Берем с минусом, т.к. мы максимизировали -sum(artificial)
        
        if optimal_value_phase1 < -1e-10:  # Если оптимальное значение < 0 (с учетом погрешности)
            # Нет допустимого решения
            solution = np.zeros(n)
            return SimplexResult(
                solution=solution,
                target_value=0,
                iterations=result_phase1[2],
                is_feasible=False,
                is_optimal=False,
                is_unbounded=False,
                is_infeasible=True,
                problem_type=problem_type,
                basic_vars=[]
            )
        
        # Фаза 2: Исключаем искусственные переменные и решаем исходную задачу
        # Удаляем искусственные переменные из таблицы
        table_phase2 = table[:-1, :].copy()  # Без строки целевой функции фазы 1
        
        # Создаем новую симплекс-таблицу для фазы 2
        new_table = np.zeros((m + 1, total_vars - n_artificial + 1))
        
        # Копируем коэффициенты без искусственных переменных
        col_idx_new = 0
        for col_idx_old in range(total_vars):
            if col_idx_old not in artificial_vars:
                new_table[:m, col_idx_new] = table_phase2[:m, col_idx_old]
                col_idx_new += 1
        
        # Копируем правые части
        new_table[:m, -1] = table_phase2[:m, -1]
        
        # Целевая функция для фазы 2
        # Преобразуем целевую функцию для исходной задачи
        if problem_type == ProblemType.MIN:
            c_ext = np.concatenate([c, np.zeros(n_slack + n_surplus + n_artificial)])
        else:  # MAX
            c_ext = np.concatenate([-c, np.zeros(n_slack + n_surplus + n_artificial)])  # Берем с минусом для приведения к минимизации
        
        # Удаляем искусственные переменные из целевой функции
        c_phase2 = np.zeros(total_vars - n_artificial)
        col_idx_new = 0
        for col_idx_old in range(total_vars):
            if col_idx_old not in artificial_vars:
                c_phase2[col_idx_new] = c_ext[col_idx_old]
                col_idx_new += 1
        
        # Добавляем целевую строку
        new_table[-1, :-1] = c_phase2
        new_table[-1, -1] = 0  # Начальное значение целевой функции
        
        # Обновляем базисные переменные (удаляем искусственные)
        basic_vars_phase2 = []
        basic_row_map = {}
        
        for i, var_idx in enumerate(basic_vars):
            if var_idx in artificial_vars:
                # Искусственная переменная в базисе - ищем замену
                # Ищем небазисную переменную с ненулевым коэффициентом в этой строке
                found_replacement = False
                for j in range(total_vars - n_artificial):
                    if abs(new_table[i, j]) > 1e-10 and j not in [v for v, _ in basic_row_map.values()]:
                        # Нашли замену
                        basic_vars_phase2.append(j)
                        basic_row_map[j] = (i, j)
                        found_replacement = True
                        break
                
                if not found_replacement:
                    # Эта строка линейно зависима, можно удалить
                    pass
            else:
                # Пересчитываем индекс переменной без учета искусственных
                new_idx = 0
                for old_idx in range(var_idx):
                    if old_idx not in artificial_vars:
                        new_idx += 1
                basic_vars_phase2.append(new_idx)
                basic_row_map[new_idx] = (i, new_idx)
        
        # Пересчитываем целевую строку для базисных переменных
        for var_idx, (row_idx, _) in basic_row_map.items():
            new_table[-1] -= new_table[-1, var_idx] * new_table[row_idx]
        
        # Обновляем total_vars для фазы 2
        total_vars = total_vars - n_artificial
        
        # Решаем фазу 2
        final_table, final_basic_vars, iterations, is_unbounded, table_history = _simplex_iterations(
            new_table, basic_vars_phase2,
            max_iterations=max_iterations,
            record_history=record_history,
            prev_iterations=result_phase1[2],
            prev_history=result_phase1[4]
        )
        
    else:
        # Нет искусственных переменных - можно решать напрямую
        # Создаем симплекс-таблицу
        table = np.zeros((m + 1, total_vars + 1))
        table[:m, :total_vars] = A_ext
        table[:m, -1] = b.copy()
        
        # Преобразуем целевую функцию
        if problem_type == ProblemType.MIN:
            c_ext = np.concatenate([c, np.zeros(n_slack + n_surplus)])
        else:  # MAX
            c_ext = np.concatenate([-c, np.zeros(n_slack + n_surplus)])  # Берем с минусом для приведения к минимизации
        
        table[-1, :total_vars] = c_ext
        table[-1, -1] = 0  # Начальное значение целевой функции
        
        # Начальный базис: slack переменные
        basic_vars = slack_vars.copy()
        
        # Пересчитываем целевую строку для базисных переменных
        for i, var_idx in enumerate(basic_vars):
            table[-1] -= table[-1, var_idx] * table[i]
        
        # Решаем задачу
        final_table, final_basic_vars, iterations, is_unbounded, table_history = _simplex_iterations(
            table, basic_vars,
            max_iterations=max_iterations,
            record_history=record_history,
        )
        total_vars = n + n_slack + n_surplus
    
    # 3. Извлекаем решение
    solution_ext = np.zeros(total_vars)
    for i, var_idx in enumerate(final_basic_vars):
        if var_idx < len(solution_ext):
            solution_ext[var_idx] = final_table[i, -1]
    
    # Берем только исходные переменные
    solution = solution_ext[:n]
    
    # Целевое значение
    # Инвертируем значения, так как в таблице хранится отрицательное значение
    if problem_type == ProblemType.MIN:
        target_value = -final_table[-1, -1]
    else:
        target_value = final_table[-1, -1]
    
    # Определяем оптимальность
    is_optimal = not is_unbounded and iterations < max_iterations
    
    # Создаем список базисных переменных (только исходные)
    basic_vars_original = []
    for var_idx in final_basic_vars:
        if var_idx < n:
            basic_vars_original.append(var_idx)
    
    
    return SimplexResult(
        solution=solution,
        target_value=target_value,
        iterations=iterations,
        is_feasible=True,
        is_optimal=is_optimal,
        is_unbounded=is_unbounded,
        is_infeasible=False,
        problem_type=problem_type,
        basic_vars=basic_vars_original,
        table_history=table_history if record_history else []
    )


# Выполняет итерации симплекс-метода.
def _simplex_iterations(
    table: np.ndarray,
    basic_vars: List[int],
    max_iterations: int,
    record_history: bool,
    prev_iterations: int = 0,
    prev_history: List[np.ndarray] = None
) -> Tuple[np.ndarray, List[int], int, bool, List[np.ndarray]]:
    m, n = table.shape
    n_vars = n - 1  # Количество переменных (без столбца RHS)
    
    iterations = prev_iterations
    table_history = prev_history.copy() if prev_history else []
    
    if record_history:
        table_history.append(table.copy())
    
    for iteration in range(max_iterations):
        # 1. Проверка оптимальности (все коэффициенты в целевой строке >= 0 для минимизации)
        optimal = True
        for j in range(n_vars):
            if table[-1, j] < -1e-10:  # Отрицательный коэффициент (с учетом погрешности)
                optimal = False
                break
        
        if optimal:
            break
        
        # 2. Выбор входящей переменной (наибольший по модулю отрицательный коэффициент)
        entering_var = -1
        max_neg = 0
        for j in range(n_vars):
            if table[-1, j] < -1e-10 and abs(table[-1, j]) > max_neg:
                max_neg = abs(table[-1, j])
                entering_var = j
        
        if entering_var == -1:
            break
        
        # 3. Выбор выходящей переменной (минимальное отношение RHS/коэффициент)
        exiting_var = -1
        min_ratio = float('inf')
        
        for i in range(m - 1):  # Все строки кроме целевой
            if table[i, entering_var] > 1e-10:  # Положительный коэффициент
                ratio = table[i, -1] / table[i, entering_var]
                if ratio >= 0 and ratio < min_ratio - 1e-10:
                    min_ratio = ratio
                    exiting_var = i
        
        # Если не нашли выходящую переменную - задача неограничена
        if exiting_var == -1:
            return table, basic_vars, iterations + iteration + 1, True, table_history
        
        # 4. Обновление базиса
        # Проверяем, что exiting_var в пределах списка
        if exiting_var < len(basic_vars):
            basic_vars[exiting_var] = entering_var
        else:
            # Добавляем новый элемент, если нужно
            while len(basic_vars) <= exiting_var:
                basic_vars.append(-1)
            basic_vars[exiting_var] = entering_var
        
        # 5. Пересчет таблицы (метод Гаусса-Жордана)
        # Нормируем ведущую строку
        pivot = table[exiting_var, entering_var]
        table[exiting_var] /= pivot
        
        # Обнуляем ведущий столбец в других строках
        for i in range(m):
            if i != exiting_var:
                factor = table[i, entering_var]
                if abs(factor) > 1e-10:
                    table[i] -= factor * table[exiting_var]
        
        iterations += 1
        
        if record_history:
            table_history.append(table.copy())
    
    if iteration >= max_iterations:
        is_unbounded = True
    else:
        is_unbounded = False  # Если дошли сюда, то неограниченность не обнаружена
    
    return table, basic_vars, iterations, is_unbounded, table_history


# Форматирует задачу в читаемый вид.
def format_problem(
    A: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    signs: List[CompareSign],
    problem_type: ProblemType
) -> str:
    lines = []
    lines.append("=" * 70)
    lines.append("LINEAR PROGRAMMING PROBLEM")
    lines.append("=" * 70)
    
    objective_parts = []
    for i, coeff in enumerate(c):
        if abs(coeff) > 1e-10:
            objective_parts.append(f"{coeff:+.4f} * x{i+1}")
    
    if not objective_parts:
        objective_str = "0"
    else:
        first = objective_parts[0]
        if first.startswith('+'):
            first = first[1:]
        objective_parts[0] = first
        objective_str = "".join(objective_parts)
    
    lines.append(f"\n{problem_type.value.upper()}: F = {objective_str}")
    
    lines.append("\n\nSUBJECT TO:")
    for i in range(len(b)):
        constraint_parts = []
        for j in range(len(c)):
            coeff = A[i, j]
            if abs(coeff) > 1e-10:
                constraint_parts.append(f"{coeff:+.4f} * x{j+1}")
        
        if constraint_parts:
            first = constraint_parts[0]
            if first.startswith('+'):
                first = first[1:]
            constraint_parts[0] = first
            constraint_str = "".join(constraint_parts)
        else:
            constraint_str = "0"
        
        sign_symbol = signs[i].value
        lines.append(f"  {constraint_str} {sign_symbol} {b[i]:.4f}")
    
    nonneg_conditions = ", ".join([f"x{i+1} >= 0" for i in range(len(c))])
    lines.append(f"\nWITH: {nonneg_conditions}")
    
    lines.append("\n" + "=" * 70)
    return "\n".join(lines)


# Форматирует симплекс-таблицу для вывода.
def format_simplex_table(table: np.ndarray, iteration: int, basic_vars: List[int] = None) -> str:
    m, n = table.shape
    
    # Определяем заголовки столбцов
    n_vars = n - 1
    headers = [f"x{i+1}" for i in range(n_vars)] + ["RHS"]
    
    rows = []
    for i in range(m-1):
        row_label = f"x{basic_vars[i]+1}" if basic_vars and i < len(basic_vars) else f"Row {i+1}"
        row = [row_label] + [f"{val:10.6f}" for val in table[i]]
        rows.append(row)
    
    rows.append(["F"] + [f"{val:10.6f}" for val in table[-1]])
    
    table_str = f"Iteration {iteration}:\n"
    header_line = "  " + "  ".join(f"{h:>10}" for h in headers)
    table_str += header_line + "\n"
    table_str += "-" * len(header_line) + "\n"
    for row in rows:
        table_str += f"{row[0]:<4}" + "  ".join(f"{cell:>10}" for cell in row[1:]) + "\n"
    
    return table_str


# Проверяет удовлетворение ограничений найденным решением.
def check_solution(
    result: SimplexResult,
    A: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    signs: List[CompareSign],
    tol: float = 1e-6
) -> Dict[str, List[bool]]:
    if not result.is_feasible:
        return {"status": "infeasible", "constraints": []}
    
    checks = {"status": "feasible", "constraints": []}
    
    for i in range(len(b)):
        left_side = np.dot(A[i], result.solution)
        right_side = b[i]
        
        satisfied = False
        if signs[i] == CompareSign.LESS_EQUAL:
            satisfied = left_side <= right_side + tol
        elif signs[i] == CompareSign.EQUAL:
            satisfied = abs(left_side - right_side) <= tol
        else:
            satisfied = left_side >= right_side - tol
        
        checks["constraints"].append({
            "index": i,
            "left_side": left_side,
            "right_side": right_side,
            "sign": signs[i].value,
            "satisfied": satisfied
        })
    
    calculated_value = np.dot(c, result.solution)
    checks["target_check"] = abs(calculated_value - result.target_value) <= tol
    
    return checks


def print_solution_report(
    result: SimplexResult,
    A: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    signs: List[CompareSign]
) -> str:
    """
    Создает полный отчет о решении.
    """
    lines = []
    
    lines.append(str(result))
    
    if result.is_feasible:
        checks = check_solution(result, A, b, c, signs)
        
        lines.append("\nCONSTRAINT VERIFICATION:")
        lines.append("-" * 40)
        
        for check in checks["constraints"]:
            lines.append(
                f"Constraint {check['index']+1}: "
                f"{check['left_side']:.6f} {check['sign']} {check['right_side']:.4f}"
            )
        
        lines.append(f"\nTarget function verification:")
        lines.append(f"  Calculated: {np.dot(c, result.solution):.6f}")
        lines.append(f"  From table: {result.target_value:.6f}")
        
        if result.table_history:
            lines.append(f"\nTable iterations recorded: {len(result.table_history)}")
    
    lines.append("=" * 70)
    return "\n".join(lines)
