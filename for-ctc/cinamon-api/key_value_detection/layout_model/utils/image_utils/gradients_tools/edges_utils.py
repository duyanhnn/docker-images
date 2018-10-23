import cv2
import numpy as np
from numpy import linalg as LA

from ...generic_utils import take_skip_array


def get_edge_index(colored_image_array, index, first_threshold,
                   second_threshold):
    return cv2.Canny(colored_image_array[:, :, index],
                     first_threshold, second_threshold)


def get_edge_index_flatten(colored_image_array, index, first_threshold,
                           second_threshold):
    return get_edge_index(colored_image_array, index, first_threshold,
                          second_threshold).flatten()


def get_blurred_contour(contour, window_size, mean_signal):
    """
    By appending window_size before and window_size after, takes mean
    """
    new_contour = list(contour)[-int((window_size-1)/2):]
    new_contour.extend(list(contour))
    new_contour.extend(list(contour)[:int((window_size-1)/2)])
    new_contour = np.array(new_contour)
    new_contour = new_contour.reshape((new_contour.shape[0], 2))
    mean_contour = np.zeros((new_contour.shape[0]-int(window_size-1), 2))
    mean_contour[:, 0] = np.correlate(new_contour[:, 0], mean_signal)
    mean_contour[:, 1] = np.correlate(new_contour[:, 1], mean_signal)
    return mean_contour


def get_normals_contour_smooth(contour_smooth, skip, debug=False):
    """Generate normal vectors
    Input:
        contour_smooth: contour point smoothed, Nx2 matrix
        skip: number of element each skip
    Ouput:
        norms: each rows is a normed vector
    """
    norm_matrix = np.array([[0, -0.5], [0.5, 0]])
    mean_contour = contour_smooth
    if debug:
        print("num mean: ", mean_contour.shape[0])
    if mean_contour.shape[0] > skip:
        skip = int(mean_contour.shape[0]/skip)
    if debug:
        print("skip: ", skip)
    # Rotate forward and rotate backward
    mean_plus_1 = np.roll(mean_contour, 1, axis=0)
    mean_minu_1 = np.roll(mean_contour, -1, axis=0)
    # Skipping
    mean_contour = take_skip_array(mean_contour, skip)
    mean_plus_1 = take_skip_array(mean_plus_1, skip)
    mean_minu_1 = take_skip_array(mean_minu_1, skip)
    # Take sample colors

    offset_1 = mean_plus_1 - mean_contour
    offset_2 = mean_contour - mean_minu_1
    offset1_norm = LA.norm(offset_1, axis=1)
    offset2_norm = LA.norm(offset_2, axis=1)
    indices = (offset1_norm != 0)*(offset2_norm != 0)
    offset_1 = offset_1[indices]/offset1_norm[indices][:, None]
    offset_2 = offset_2[indices]/offset2_norm[indices][:, None]
    normed_offset = offset_1+offset_2
    norms = np.transpose(np.dot(norm_matrix, np.transpose(normed_offset)))
    return mean_contour[indices, :], norms


def get_normals_contour(contour, window_size=5, skip=6):
    """
    Input:
        contour: Nx2 points
        window_size: mean size
        skip: skip range
    Output:
        normals: Nx2 points
    Merging blur contour and get normal contour smooth"""
    if window_size % 2 == 0:
        window_size += 1
    mean_signal = np.ones(window_size)/window_size
    mean_contour = get_blurred_contour(contour, window_size, mean_signal)
    return get_normals_contour_smooth(mean_contour, skip)
