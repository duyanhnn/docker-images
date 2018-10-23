import numpy as np


def plot_gradient(imggray, ax):
    """Plot gradient of the image"""
    p = imggray.astype('int8')
    w, h = p.shape[1], p.shape[0]
    x, y = np.mgrid[0:w, 0:h]

    dy, dx = np.gradient(p)
    skip = (slice(None, None, 5), slice(None, None, 5))

    ax.quiver(x[skip], y[skip], dx[skip].T, dy[skip].T,
              scale=15,
              scale_units='inches')

    ax.set(aspect=1, title='Quiver Plot')


def plot_gdc(imggray, ax, gdc):
    """Plot gradient of the image"""
    p = imggray.astype('int8')
    w, h = p.shape[1], p.shape[0]
    x, y = np.mgrid[0:w, 0:h]

    dy, dx = gdc[:, :, 1], gdc[:, :, 0]
    skip = (slice(None, None, 5), slice(None, None, 5))

    ax.quiver(x[skip], y[skip], dx[skip].T, dy[skip].T,
              scale=10,
              scale_units='inches')

    ax.set(aspect=1, title='Quiver Plot')
