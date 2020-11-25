import math
import numpy as np 
import numba
from numba import cuda
import time
# comstomer functions
from cf import pts_generator, visualize



@cuda.jit
def rbf_evaluation_cuda_temp(sources, targets, weights, result_temp):
    local_result = cuda.shared.array((SX, SY), numba.float32)
    local_targets = cuda.shared.array((SX, 3), numba.float32)
    local_sources = cuda.shared.array((SY, 3), numba.float32)
    local_weights = cuda.shared.array(SY, numba.float32)
    
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    
    by = cuda.blockIdx.y

    px, py = cuda.grid(2)
    
    if px >= targets.shape[0]:
        return

    # At first we are loading all the targets into the shared memory
    # We use only the first column of threads to do this.
    
    if ty == 0:
        for index in range(3):
            local_targets[tx, index] = targets[px, index]
    
    # We are now loading all the sources and weights.
    # We only require the first row of threads to do this.
    
    if tx == 0:
        for index in range(3):
            local_sources[ty, index] = sources[py, index]
        local_weights[ty] = weights[py]
        
    # Let us now sync all threads
    # this mean we wait for all previous kernel launches to finish executing
    cuda.syncthreads()
    
    # Now compute the interactions
    
    squared_diff = numba.float32(0)
    
    for index in range(3):
        squared_diff += (local_targets[tx, index] - local_sources[ty, index])**2
    local_result[tx, ty] = math.exp(-squared_diff / ( numba.float32(2) * numba.float32(sigma)**2)) * local_weights[ty]
    
    cuda.syncthreads()
    
    # Now sum up all the local results
    
    if ty == 0:
        res = numba.float32(0)
        for index in range(SY):
            res += local_result[tx, index]
        result_temp[px, by] = res    

@cuda.jit
def result_temp_sum(result_temp, result):
    local_values = cuda.shared.array(col_blocks, numba.float32)
    
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y

    px, _ = cuda.grid(2)
    
    if ty == 0:
        res = numba.float32(0)
        for index in range(col_blocks):
            res += result_temp[px, index]
    result[px] = res
    
    # if ty == 0:
    #     for index in range(col_blocks):
    #         local_values[index] = result_temp[px, index]
    
    # cuda.syncthreads()

    # if ty == 0:
    #     res = numba.float32(0)
    #     for index in range(col_blocks):
    #         res += local_values[index]
    #     result[px] = res

# external configuration
sigma = .1
npoints = 200
nsources = 800
# data based on sources and targets
targets, sources, weights = pts_generator(npoints, nsources)
ntargets = targets.shape[0]

# equivalent
# blocks = (targets.shape[0] + SX - 1) // SX
SX = 16
SY = 32
row_blocks = (ntargets + SX - 1) // SX
col_blocks = (nsources + SY - 1) // SY

t0 = time.time()
# in this implementation we use result_temp as an intermediate storage of the each local result
result_temp = numba.cuda.device_array((ntargets, col_blocks), dtype=np.float32)
targets_cuda = numba.cuda.to_device(targets)
sources_cuda = numba.cuda.to_device(sources)
weights_cuda = numba.cuda.to_device(weights)
result = numba.cuda.device_array((ntargets), dtype=np.float32)
#result = np.zeros(ntargets, dtype=np.float32)


# cuda_kernal[blockspergrid, threadsperblock](a, b, c, ...)
# rbf_evaluation_cuda[(nblocks, 1), (SX, SY)](sources.astype('float32'), targets.astype('float32'), weights.astype('float32'), result)
# rbf_evaluation_cuda_temp[(row_blocks, col_blocks), (SX, SY)](sources.astype('float32'), targets.astype('float32'), weights.astype('float32'), result_temp)

rbf_evaluation_cuda_temp[(row_blocks, col_blocks), (SX, SY)](sources_cuda, targets_cuda, weights_cuda, result_temp)
#result_temp = result_temp.copy_to_host()
#result = np.sum(result_temp, axis=1)
result_temp_sum[(ntargets, 1), (1, col_blocks)](result_temp, result)
result = result.copy_to_host()
t1 = time.time()
t = t1-t0
print(t)
visualize(result, npoints)