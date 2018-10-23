from __future__ import print_function
from __future__ import division
import numpy as np
from numpy import clip, minimum, array, prod, amax, ones

from scipy.ndimage import filters, morphology, interpolation
from scipy import stats


def ocropus_binarize_code(image, threshold=0.5, perc=80,
                          filter_range=20, bignore=0.1,
                          escale=1.0, lo=5, hi=90,
                          zoom=0.5):
    """ Ocropus nlbin repackaged
    Input:
        threshold: determine lightness
        perc: for percentile filter
        filter_range: for percentile filter
        bignore: ignore this much of the border for threshold estimation
        escale: scale for estimating a mask over text region
        lo: percentitle for black estimation
        hi: percentitle for white estimation
        zoom: for page background estimation
    """
    extreme = (np.sum(image < 0.05) + np.sum(image > 0.95)
               ) * 1.0 / prod(image.shape)
    if extreme > 0.95:
        flat = image.astype('float64')/255.0
    else:
        image = image.copy().astype('float64')/255.0
        m = interpolation.zoom(image, zoom)
        m = filters.percentile_filter(m, perc, size=(filter_range, 2))
        m = filters.percentile_filter(m, perc, size=(2, filter_range))
        w, h = minimum(array(image.shape), array(m.shape))
        m = interpolation.zoom(m, 1.0 / zoom)
        flat = clip(image[:w, :h] - m[:w, :h] + 1, 0, 1).astype('float64')/255

    d0, d1 = flat.shape
    o0, o1 = int(bignore * d0), int(bignore * d1)
    est = flat[o0:d0 - o0, o1:d1 - o1]
    if escale > 0:
        e = escale
        v = est - filters.gaussian_filter(est, e * 20.0)
        v = filters.gaussian_filter(v ** 2, e * 20.0) ** 0.5
        v = (v > 0.3 * amax(v))
        v = morphology.binary_dilation(v, structure=ones((int(e * 50), 1)))
        v = morphology.binary_dilation(v, structure=ones((1, int(e * 50))))
        est = est[v]

    lo = stats.scoreatpercentile(est.ravel(), lo)
    hi = stats.scoreatpercentile(est.ravel(), hi)
    flat -= lo
    flat /= (hi - lo)
    flat = clip(flat, 0, 1).astype('float64')

    bin = 255 * (flat > threshold)
    bin = bin.astype('uint8')
    return bin
