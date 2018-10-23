from __future__ import print_function
from __future__ import division

import numpy as np

from ..bbox_tools.bbox_operations import get_width_height_bounding_boxs
from ..local_imutils import (
    connected_component_analysis_sk_with_labels as cca_sk)
from ...generic_utils import sorted_index


def customized_filter_width_height_bboxs(
        bboxs,
        upper_bound_height_ratio,
        lower_bound_height_ratio,
        lower_bound_width_ratio):
    """
    Combined filtering like in the papers, returns bboxs and indices of labeling
    Args:
        bboxs: bounding boxes of connected component
        upper_bound_height_ratio: upperbound comparing to max occurrences H
        lower_bound_height_ratio: lowerbound comparing to max occurrences H
        lower_bound_width_ratio: lowerbound comparing to max occurrences H
    Output:
        bboxs_filtereds: filtered set of bboxs
        indices: index in original bounding boxes list
    """
    widths, heights = get_width_height_bounding_boxs(bboxs)
    unique_heights = set(heights)
    countings = [heights.count(height) for height in unique_heights]
    sorted_indices = sorted_index(countings, True)
    max_height_occurrences = sorted_indices[0]
    upper_bound_height = max_height_occurrences*upper_bound_height_ratio
    lower_bound_height = max_height_occurrences*lower_bound_height_ratio
    lower_bound_width = max_height_occurrences*lower_bound_width_ratio

    indices = [i for i in range(len(heights))
               if heights[i] > lower_bound_height and
               heights[i] < upper_bound_height and
               widths[i] > lower_bound_width]
    return bboxs[indices], indices


def word_detector(bin_image,
                  upper_bound_height_ratio=3,
                  lower_bound_height_ratio=0.25,
                  lower_bound_width_ratio=0.25,
                  debug=False):
    """
    Word detector in gatos et al paper of dewarping.
    """
    # Step 1, apply connected component labeling
    labels, bboxs = cca_sk(bin_image)
    if len(bboxs == 0):
        if debug:
            print("Warning 'word_detector': no ccs found")
        return bin_image
    # Step 2, calculate the histogram with the heights of all detected ccs
    # Step 3, removing bboxs
    bboxs_filtered, indices = customized_filter_width_height_bboxs(
        bboxs, upper_bound_height_ratio, lower_bound_height_ratio,
        lower_bound_width_ratio)
    # Step 4, Apply horizontal smoothing, with threshold H folloed by a cc
    # labeling in order to detect word
