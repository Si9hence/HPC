import numpy as np
import numba
from numba import njit, prange
numba.config.NUMBA_DEFAULT_NUM_THREADS = 4

@njit(parallel=True)
def diffusion_iteration_njit_par_fix(un, constant_indices=np.array([])):

    #initialization
    res = np.empty(un.shape)
    #change range to prange to let numba know where to optimize
    for irow in prange(un.shape[0]):
        #the prange function is only applied to the outer for-loop to aviod nested parallelisation 
        for icol in range(un.shape[1]):
            #for points at boundaries keep their values fixed
            if irow == 0 or icol == 0 or irow == un.shape[0] - 1 or icol == un.shape[1] - 1:
                res[irow][icol] = un[irow][icol]
            else:
                #else take the average of adjacent points
                res[irow][icol] = (un[irow + 1][icol] + un[irow - 1][icol] + un[irow][icol + 1] + un[irow][icol - 1]) / 4
     
    for i in range(np.shape(idx_fix)):
         res[idx_fix[i][0]][idx_fix[i][1]] = un[idx_fix[i][0]][idx_fix[i][1]]

    return res


import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from IPython.display import HTML

idx_fix = []
#the index of fixed points are separated by 10 unit an spread uniformly over the grid.
for i in range(0, 500, 10):
    for j in range(0, 500, 10):
        idx_fix.append([i, j])

idx_fix = np.array(idx_fix)

fps = 30
n_seconds = 8

# we set the grid size = 500 and the highest temperature of  row and col with index (100: 400) is set to be 200 and decay linearly
# actually the heat distribution could be more precise if Boltzmanm or normal distribution
grid_size_vis = 500
ini_distribution = np.zeros((grid_size_vis, grid_size_vis))
for irow in range(100, 400):
    for icol in range(100, 400):
        ini_distribution[irow][icol] = 200 - np.sqrt(abs((irow-250))**2 + abs(icol - 250)**2)

temp = ini_distribution.copy()

# First set up the figure, the axis, and the plot element we want to animate
fig = plt.figure()
ax = plt.axes(xlim=(0, 500), ylim=(0, 500))
im = plt.imshow(ini_distribution,interpolation='none')
fig.colorbar(im)

# initialization function: plot the background of each frame
def init():
    im.set_data(ini_distribution)
    return [im]

# animation function.  This is called sequentially
def animate(i):
    temp = im.get_array()
    temp = diffusion_iteration_njit_par_fix(temp, idx_fix)   
    im.set_array(temp)
    return [im]

anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=fps*n_seconds, interval=1000/fps, blit=True)

plt.show()