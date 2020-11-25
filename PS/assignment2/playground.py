import math
import numpy as np 
import numba
from numba import cuda

# comstomer functions
from cf import pts_generator, visualize



@cuda.jit
def rbf_evaluation_cuda_temp(sources, targets, weights, result):
    local_result = cuda.shared.array((SX, SY), numba.float32)
    local_targets = cuda.shared.array((SX, 3), numba.float32)
    local_sources = cuda.shared.array((SY, 3), numba.float32)
    local_weights = cuda.shared.array(SY, numba.float32)
    
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    
    px, py = cuda.grid(2)
    
    if ty:
        result[ty] = py


# external configuration
sigma = .1
npoints = 400
nsources = 32
rand = np.random.RandomState(0)
# data based on sources and targets
targets, sources, weights = pts_generator(npoints, nsources)
mtargets = targets.shape[0]

# equivalent
# blocks = (targets.shape[0] + SX - 1) // SX
SX = 16
SY = 32
row_blocks = math.ceil(mtargets / SX)
col_blocks = math.ceil(nsources / SY)


# in this implementation we use result_temp as an intermediate storage of the each local result
result_temp = numba.cuda.device_array((row_blocks, SX), dtype=np.float32)
result = np.zeros(mtargets, dtype=np.float32)


# cuda_kernal[blockspergrid, threadsperblock](a, b, c, ...)
# rbf_evaluation_cuda[(nblocks, 1), (SX, SY)](sources.astype('float32'), targets.astype('float32'), weights.astype('float32'), result)
rbf_evaluation_cuda_temp[(row_blocks, col_blocks), (SX, SY)](sources.astype('float32'), targets.astype('float32'), weights.astype('float32'), result)



# visualize(result, npoints)