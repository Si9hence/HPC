import numpy as np
import numba
from cf import pts_generator, visualize

sigma = .1

@numba.njit(parallel=True)
def rbf_evaluation(sources, targets, weights, result):
    """Evaluate the RBF sum."""
    
    n = len(sources)
    m = len(targets)
        
    result[:] = 0
    for index in numba.prange(m):
        result[index] = np.sum(np.exp(-np.sum(np.abs(targets[index] - sources)**2, axis=1) / (2 * sigma**2)) * weights)

sigma = .1
npoints = 400
nsources = 800
# data based on sources and targets
targets, sources, weights = pts_generator(npoints, nsources)

result = np.zeros(len(targets), dtype=np.float64)


rbf_evaluation(sources, targets, weights, result)
    
visualize(result, npoints)