from __future__ import print_function
from __future__ import division

import numpy as np
from ...generic_utils import sorted_index
from scipy.signal import correlate2d


def crla_correlate(binary_image, constraint_value, horizontal=True,
                   debug=False):
    """
    Constrained runlength algorithm, using correlation
    Args:
        binary_image: binarized image
        constraint_value: constrain length
        horizontal: True if horizontal, False for vertical
    Output:
        binary_image_processed
    """
    # Padding binary image with 1s
    total_affected = constraint_value + 2
    kernel_filters_pre = []
    kernel_filters_pos = []
    for i in range(constraint_value):
            if i == 0:
                continue
            pre_size = i + 2
            pos_size = total_affected-pre_size + 1
            kernel_filters_pre.append(np.ones((1, pre_size)))
            kernel_filters_pos.append(np.ones((1, pos_size)))
    if total_affected < 3:
        return binary_image
    if horizontal:
        paddings = np.ones((binary_image.shape[0], 1))
        binary_image = np.hstack([paddings, binary_image])
        binary_image = np.hstack([binary_image, paddings])
        # Mirror...
    else:
        kernel_filters_pre = [np.transpose(kernel)
                              for kernel in kernel_filters_pre]
        kernel_filters_pos = [np.transpose(kernel)
                              for kernel in kernel_filters_pos]
        paddings = np.ones((1, binary_image.shape[1]))
        binary_image = np.vstack([paddings, binary_image])
        binary_image = np.vstack([binary_image, paddings])
        # Mirror...
    binary_images = []
    flip_func = np.fliplr
    if horizontal:
        flip_func = np.flipud
    for i, kernel_pre in enumerate(kernel_filters_pre):
        binary_image_pre = correlate2d(binary_image,
                                       kernel_pre, 'same')
        binary_image_pos = flip_func(correlate2d(flip_func(binary_image),
                                                 kernel_filters_pos[i],
                                                 'same'))
        binary_images.append(binary_image_pre*binary_image_pos)
    binary_image = np.sum(np.array(binary_images), axis=0) != 0
    return binary_image*255


def clra_npwhere(binary_image, constraint_value, horizontal=True, debug=False):
    """
    Constrained runlength algorithm, using npwhere
    Args:
        binary_image: binarized image
        constraint_value: constrain length
        horizontal: True if horizontal, False for vertical
    Output:
        binary_image_processed
    """
    one_pos = np.where(binary_image != 0)
    # Sort after the same indices first then
    binary_image = np.zeros_like(binary_image)
    main_index = 0
    if not horizontal:
        main_index = 1
    one_pos_main = list(set(one_pos[main_index]))
    # Only loop throught set of main
    for line in one_pos_main:
        # Get all the position that have this value
        positions = np.where(one_pos[main_index] == line)
        sub_pos = one_pos[1-main_index][positions]
        sorted_pos_indices = sorted_index(sub_pos)
        if len(sorted_pos_indices) == 1:
            current_sub_value = sub_pos[sorted_pos_indices[0]]
            if current_sub_value < constraint_value:
                # Fill from the begining of the line until that point
                if main_index == 0:
                    binary_image[line, 0:current_sub_value] = 1
                else:
                    binary_image[0:current_sub_value, line] = 1
        for i in range(len(sorted_pos_indices)-1):
            current_sub_value = sub_pos[sorted_pos_indices[i]]
            next_sub_value = sub_pos[sorted_pos_indices[i+1]]
            if (next_sub_value - current_sub_value) < constraint_value + 1:
                if main_index == 0:
                    binary_image[line, current_sub_value:next_sub_value] = 1
                else:
                    binary_image[current_sub_value:next_sub_value, line] = 1
    return binary_image*255
