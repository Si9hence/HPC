import numpy as np 
import numba
from numba import cuda


n = 200
u = np.random.RandomState(0).rand(n*n).astype("float64")
res = np.zeros(u.shape, dtype=np.float64)

tpb = 1024
nob = (len(u) + 1023) // 1024

@cuda.jit
def mat_A_cuda(u, res):
    px = cuda.grid(1)
    # u_{i,j} = x_{jN+i}
    # case x == 0
    if px % n == 0:
        res[px] = u[px]
    # case x == n - 1
    elif px % n == n - 1:
        res[px] = u[px] 
    # case y == 0
    elif px < n:
        res[px] = u[px]
    # case y == n - 1
    elif px // n == n - 1:
        res[px] = u[px]
    else:
        res[px] = (4*u[px] - u[px-1] - u[px+1] - u[px-n] - u[px+n]) * (n-1)**2

    

mat_A_cuda[nob, tpb](u, res)
