
# First set up the figure, the axis, and the plot element we want to animate
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.animation as animation

import numba
from numba import njit, prange
numba.config.NUMBA_DEFAULT_NUM_THREADS = 4

@njit(parallel=True)
def diffusion_iteration_njit_par(un):

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
     
    return res

fps = 30
n_seconds = 8

grid_size_vis = 500
ini_distribution = np.zeros((grid_size_vis, grid_size_vis))
for irow in range(100, 400):
    for icol in range(100, 400):
        ini_distribution[irow][icol] = 200 - np.sqrt(abs((irow-250))**2 + abs(icol - 250)**2)

# First set up the figure, the axis, and the plot element we want to animate
fig = plt.figure()
ax = plt.axes(xlim=(0, 500), ylim=(0, 500))
#line, = ax.plot([], [], lw=2)
im = plt.imshow(ini_distribution,interpolation='none')
fig.colorbar(im)
temp = ini_distribution.copy()
# initialization function: plot the background of each frame

def init():
    im.set_data(ini_distribution)
    return [im]

# animation function.  This is called sequentially
def animate(i):
    temp = im.get_array()
    temp = diffusion_iteration_njit_par(temp)    # exponential decay of the values
    im.set_array(temp)
    return [im]

anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=fps*n_seconds, interval=1000/fps, blit=True)


plt.show()