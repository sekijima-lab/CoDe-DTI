import numpy as np

def solve_cpu(a, b):
    np.copyto(b, np.linalg.solve(a, b))

def solve(a, b):
    if not isinstance(a, np.ndarray) or not isinstance(b, np.ndarray):
        raise ValueError("Only CPU arrays are validated")
    solve_cpu(a, b)
