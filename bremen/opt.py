import numpy as np
import chainer


def solve_cpu(a, b):
    x = np.linalg.solve(a, b)
    np.copyto(b, x)


def solve(a, b):
    if chainer.cuda.get_array_module(a) == np:
        solve_cpu(a, b)
    else:
        solve_gpu(a, b)

try:
    if chainer.cuda.available:
        from cupy import cuda

        if cuda.cusolver_enabled:
            from opt_gpu import solve_gpu
except:
    pass
