import cv2
import numpy as np


def get_line_averaging_filter(l, theta):
    """Get line averaging filter given length and theta
    """
    x_start = int(-l*np.cos(theta)*10+l/2)
    y_start = int(-l*np.sin(theta)*10+l/2)
    x_end = int(l*np.cos(theta)*10+l/2)
    y_end = int(l*np.sin(theta)*10+l/2)
    kernel = np.zeros((l, l), dtype=np.uint8)
    cv2.line(kernel, (x_start, y_start), (x_end, y_end), int(255), 2,
             lineType=cv2.LINE_AA)
    return kernel.astype(np.float)/255/l/2
