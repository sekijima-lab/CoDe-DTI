"""Historical CUDA 8 solver is retained on master, not in this CPU runtime."""
def solve_gpu(a, b):
    raise NotImplementedError("CUDA is not validated; use master for the historical GPU implementation")
