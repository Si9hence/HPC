import numpy as np
from scipy.sparse import coo_matrix

def discretise_poisson(N):
    """Generate the matrix and rhs associated with the discrete Poisson operator."""
    
    nelements = 5 * N**2 - 16 * N + 16
    
    row_ind = np.empty(nelements, dtype=np.float64)
    col_ind = np.empty(nelements, dtype=np.float64)
    data = np.empty(nelements, dtype=np.float64)
    
    f = np.empty(N * N, dtype=np.float64)
    
    count = 0
    for j in range(N):
        for i in range(N):
            if i == 0 or i == N - 1 or j == 0 or j == N - 1:
                row_ind[count] = col_ind[count] = j * N + i
                data[count] =  1
                f[j * N + i] = 0
                count += 1
                
            else:
                row_ind[count : count + 5] = j * N + i
                col_ind[count] = j * N + i
                col_ind[count + 1] = j * N + i + 1
                col_ind[count + 2] = j * N + i - 1
                col_ind[count + 3] = (j + 1) * N + i
                col_ind[count + 4] = (j - 1) * N + i
                                
                data[count] = 4 * (N - 1)**2
                data[count + 1 : count + 5] = - (N - 1)**2
                f[j * N + i] = 1
                
                count += 5
                                                
    return coo_matrix((data, (row_ind, col_ind)), shape=(N**2, N**2)).tocsr(), f



from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from scipy.sparse.linalg import spsolve

n = 200
u = np.random.RandomState(0).rand(n*n, 1).astype("float64")

A, f = discretise_poisson(n)
res = A*u
# sol = spsolve(A, f)

# u = sol.reshape((N, N))

# fig = plt.figure(figsize=(8, 8))
# ax = fig.gca(projection='3d')
# ticks= np.linspace(0, 1, N)
# X, Y = np.meshgrid(ticks, ticks)
# surf = ax.plot_surface(X, Y, u, antialiased=False, cmap=cm.coolwarm)
# plt.show()

import numpy as np 
import numba
from numba import cuda


n = 200
u = np.random.RandomState(0).rand(n*n).astype(np.float64)
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

aa = A*u
print(sum(res == A*u))
err = []
for i in range(len(res)):
    if res[i] != aa[i]:
        err.append([i, aa[i]-res[i]])


rel_error = np.linalg.norm(res - aa, np.inf) / np.linalg.norm(aa, np.inf)
print(f"Error: {round(rel_error, 2)}.")