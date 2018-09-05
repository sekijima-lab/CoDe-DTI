import numpy
from numpy import linalg

import cupy
from cupy import cuda

if cuda.cusolver_enabled:
    from cupy.cuda import cublas
    from cupy.cuda import device
    from cupy.cuda import cusolver

def solve_gpu(a, b):
    if not cuda.cusolver_enabled:
        raise RuntimeError('Current cupy only supports cusolver in CUDA 8.0')

    n = len(a)
    handle = device.get_cusolver_handle()
    dev_info = cupy.empty(1, dtype=numpy.int32)
    if a.dtype.char == 'f':
        buffersize = cusolver.spotrf_bufferSize(
            handle, cublas.CUBLAS_FILL_MODE_UPPER, n, a.data.ptr, n)
        workspace = cupy.empty(buffersize, dtype=numpy.float32)
        cusolver.spotrf(
            handle, cublas.CUBLAS_FILL_MODE_UPPER, n, a.data.ptr, n,
            workspace.data.ptr, buffersize, dev_info.data.ptr)
    elif a.dtype.char == 'd':
        buffersize = cusolver.dpotrf_bufferSize(
            handle, cublas.CUBLAS_FILL_MODE_UPPER, n, a.data.ptr, n)
        workspace = cupy.empty(buffersize, dtype=numpy.float64)
        cusolver.dpotrf(
            handle, cublas.CUBLAS_FILL_MODE_UPPER, n, a.data.ptr, n,
            workspace.data.ptr, buffersize, dev_info.data.ptr)
    else:
        raise RuntimeError('unsupported dtype')
    status = int(dev_info[0])
    if status > 0:
        raise linalg.LinAlgError(
            'The leading minor of order {} '
            'is not positive definite'.format(status))
    elif status < 0:
        raise linalg.LinAlgError(
            'Parameter error')

    if a.dtype.char == 'f':
        cusolver.spotrs(
            handle, cublas.CUBLAS_FILL_MODE_UPPER, n, 1, a.data.ptr, n, b.data.ptr, n, dev_info.data.ptr)
    else:
        cusolver.dpotrs(
            handle, cublas.CUBLAS_FILL_MODE_UPPER, n, 1, a.data.ptr, n, b.data.ptr, n, dev_info.data.ptr)
    if status < 0:
        raise linalg.LinAlgError(
            'Parameter error')