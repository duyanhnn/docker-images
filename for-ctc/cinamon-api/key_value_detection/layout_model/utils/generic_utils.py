import numpy as np
from numpy import prod


def frange(x, y, jump=1):
    if jump > 0:
        while(x < y):
            yield x
            x += jump
    elif jump < 0:
        while(x > y):
            yield x
            x += jump


def indexing_with_clipping_v2(arr, indices, clipping_value=0):
    """ Copy from stack overflow, fucking useful"""
    indices[indices >= arr.shape] = clipping_value
    return arr[indices[:, 0], indices[:, 1]]


def indexing_with_bounding_2d(arr, indices):
    """ Copy from stack overflow, fucking useful"""
    indices_shape = indices.shape
    indices = indices.flatten()
    underbound = np.zeros(prod(indices_shape))
    indices = np.maximum(indices, underbound)
    indices = indices.reshape(indices_shape)
    upperbound_y = np.ones(indices.shape[0])*(arr.shape[0]-1)
    indices[:, 0] = np.minimum(indices[:, 0], upperbound_y)
    upperbound_x = np.ones(indices.shape[0])*(arr.shape[1]-1)
    indices[:, 1] = np.minimum(indices[:, 1], upperbound_x)
    indices = indices.astype('int')

    return arr[indices[:, 0], indices[:, 1]]


def sorted_index(arr, inverse=False):
    indices = sorted(range(len(arr)), key=lambda k: arr[k])
    if inverse:
        indices = list(reversed(indices))
    return indices


def skip_array(arr, steps):
    """
    Input:
        arr: array
        steps: step to skip
    Output:
        Skipped array, at these element
    First dimmension only
    """
    mask = np.ones(arr.shape[0], dtype=bool)
    mask[::steps] = 0
    return arr[mask]


def take_skip_array(arr, steps):
    """
    Input:
        arr: array
        steps: step to skip
    Output:
        Skipped array, takes only at step elemetns
    First dimmension only
    """
    mask = np.zeros(arr.shape[0], dtype=bool)
    mask[::steps] = 1
    return arr[mask]
