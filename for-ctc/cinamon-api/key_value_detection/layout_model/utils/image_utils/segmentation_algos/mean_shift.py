from numpy import array
import cv2


def mean_shift(img, iteration=5):
    """
    Meanshift algorithm (see wiki)
    input:
        img: colored or grayscale image
    output:
        mean_shifted img
    """
