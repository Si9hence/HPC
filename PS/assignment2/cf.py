import numpy as np

def pts_generator(npoints=400, nsources=50):
    # generate random data
    plot_grid = np.mgrid[0:1:npoints * 1j, 0:1:npoints * 1j]

    targets_xy = np.vstack((plot_grid[0].ravel(),
                            plot_grid[1].ravel(),
                            np.zeros(plot_grid[0].size))).T
    targets_xz = np.vstack((plot_grid[0].ravel(),
                            np.zeros(plot_grid[0].size),
                            plot_grid[1].ravel())).T
    targets_yz = np.vstack((np.zeros(plot_grid[0].size),
                        plot_grid[0].ravel(),
                        plot_grid[1].ravel())).T

    targets = np.vstack((targets_xy, targets_xz, targets_yz))

    rand = np.random.RandomState(0)

    # We are picking random sources
    sources = rand.rand(nsources, 3)

    # generate random weights
    weights = rand.rand(len(sources))
    return targets, sources, weights


from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

def visualize(result, npoints):
    """A helper function for visualization"""
    
    result_xy = result[: npoints * npoints].reshape(npoints, npoints).T
    result_xz = result[npoints * npoints : 2 * npoints * npoints].reshape(npoints, npoints).T
    result_yz = result[2 * npoints * npoints:].reshape(npoints, npoints).T

    fig = plt.figure(figsize=(20, 20))    

    ax = fig.add_subplot(1, 3, 1)   
    _ = ax.imshow(result_xy, extent=[0, 1, 0, 1], origin='lower')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    ax = fig.add_subplot(1, 3, 2)   
    _ = ax.imshow(result_xz, extent=[0, 1, 0, 1], origin='lower')
    ax.set_xlabel('x')
    ax.set_ylabel('z')

    ax = fig.add_subplot(1, 3, 3)   
    _ = ax.imshow(result_yz, extent=[0, 1, 0, 1], origin='lower')
    ax.set_xlabel('y')
    ax.set_ylabel('z')

