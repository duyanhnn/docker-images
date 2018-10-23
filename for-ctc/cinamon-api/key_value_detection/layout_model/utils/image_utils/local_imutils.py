from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import numpy as np
import re
import cv2
import os
import subprocess
import skimage.morphology

from PIL import Image
from scipy import ndimage
from pylab import (array, uint8)
from .phase_one.dummy_rules_bbox import (dummy_hard_code_rule_big_area_height,
                                         dummy_hard_code_rule_height,
                                         dummy_filter_bbox_height_width,
                                         dummy_get_textual_bboxs)

from .bboxs_tools.bbox_operations import (
    get_area_bounding_box,
    get_minimum_bounding_rect,
    get_aspect_ratio_bounding_box,
    get_boundingbox_stats, get_boundingbox_tl_lr,
    check_intersect_interval,
    check_intersect_x, check_intersect_y,
    check_empty_interval,
    get_x_y_start_stop, check_bbox_contains_each_other,
    count_merged_lines, crop_bbox_to_file,
    sort_bbox_left_to_right, get_objs_bboxs)


def get_max_density_and_index(array_imggray_invert):
    max_density_arg = np.argmax(array_imggray_invert)
    max_density = array_imggray_invert[
        int(max_density_arg/array_imggray_invert.shape[1]),
        max_density_arg - int(
            max_density_arg/array_imggray_invert.shape[1]
        )*array_imggray_invert.shape[1]]
    return max_density_arg, max_density


def threshold_on_max(image_gray, ratio):
    max_density_arg, max_density = get_max_density_and_index(image_gray)
    ret, array_imggray_invert_threshold = cv2.threshold(
        image_gray, ratio*max_density, max_density, cv2.THRESH_BINARY)
    return array_imggray_invert_threshold


def imresize(im, sz):
    """ Resize an image array using PIL. """
    pil_im = Image.fromarray(uint8(im))
    return array(pil_im.resize(sz))


def imresize_maxdim(im, max_dim=1000):
    """Resize image so that its max dimmension is <= max_dim"""
    shape = im.shape[:2]
    if max(shape) > max_dim:
        scale = max(shape)/max_dim
        im2 = cv2.resize(im, (int(im.shape[1]/scale),
                              int(im.shape[0]/scale)))
    else:
        im2 = im
    return im2



def check_projection_merge_line(rects, types):
    """
    inputs:
        rects: all the rect bounding box
    """
    lines = []
    lines_y_range = []
    for index, rect in enumerate(rects):
        if types[index] == 0:
            y_rect_start = rect[3][1]
            y_rect_stop = rect[0][1]
            is_intersect = False
            for i, line_range in enumerate(lines_y_range):
                is_intersect2, start_new, stop_new = check_intersect_interval(
                    line_range,
                    (y_rect_start, y_rect_stop)
                )
                if is_intersect2:
                    lines[i].append(rect)
                    lines_y_range[i] = (start_new, stop_new)
                    is_intersect = True
                    break
            if not is_intersect:
                lines.append([rect])
                lines_y_range.append((y_rect_start, y_rect_stop))

    return lines, lines_y_range


def check_condition_mergable_bbox_x_intersect(labels, bbox_a, bbox_b):
    """
    """
    x_start_a, x_stop_a, y_start_a, y_stop_a = get_x_y_start_stop(bbox_a)
    x_start_b, x_stop_b, y_start_b, y_stop_b = get_x_y_start_stop(bbox_b)
    # Separated y, x intersected bbox:
    if x_start_a > x_start_b:
        bbox_a, bbox_b = bbox_b, bbox_a
    x_start_interval = x_stop_a
    x_stop_interval = x_start_b
    y_start_interval = min(y_start_a, y_start_b)
    y_stop_interval = max(y_stop_a, y_stop_b)
    if check_empty_interval(labels, x_start_interval,
                            x_stop_interval, y_start_interval,
                            y_stop_interval):
        return True
    return False


def check_condition_mergable_bbox_y_intersect(labels, bbox_a, bbox_b):
    """
    """
    x_start_a, x_stop_a, y_start_a, y_stop_a = get_x_y_start_stop(bbox_a)
    x_start_b, x_stop_b, y_start_b, y_stop_b = get_x_y_start_stop(bbox_b)
    # Separated y, x intersected bbox:
    if y_start_a > y_start_b:
        bbox_a, bbox_b = bbox_b, bbox_a
    y_start_interval = y_stop_a
    y_stop_interval = y_start_b
    x_start_interval = min(x_start_a, x_start_b)
    x_stop_interval = max(x_stop_a, x_stop_b)
    if check_empty_interval(labels, x_start_interval,
                            x_stop_interval, y_start_interval,
                            y_stop_interval):
        return True
    return False


def group_bbox_non_label_overlapping(bboxs, labels_check):
    """
    Return: group of list of bboxs
    """
    groups_bboxs = []
    groups_bboxs = [[bboxs[0]]]
    for bbox in bboxs[1:]:
        list_intersect = []
        for j, group_of_bboxs in enumerate(groups_bboxs):
            group_bbox = get_minimum_bounding_rect(group_of_bboxs)
            intersect_x = check_intersect_x(bbox, group_bbox)
            intersect_y = check_intersect_y(bbox, group_bbox)

            if intersect_x and intersect_y:
                list_intersect.append(j)
                continue
            if intersect_x:
                if check_condition_mergable_bbox_x_intersect(labels_check,
                                                             bbox, group_bbox):
                    list_intersect.append(j)
                    continue
            if intersect_y:
                if check_condition_mergable_bbox_y_intersect(labels_check,
                                                             bbox, group_bbox):
                    list_intersect.append(j)
                    continue

        indices = list_intersect
        print(len(indices))
        if len(indices) == 1:
            groups_bboxs[indices[0]].append(bbox)
        elif len(indices) > 1:
            for j in indices[1:]:
                groups_bboxs[indices[0]].extend(groups_bboxs[j])
            for j in reversed(sorted(indices[1:])):
                print("popping:", j)
                groups_bboxs.pop(j)
        else:
            print("Appending new groups")
            groups_bboxs.append([bbox])
    return groups_bboxs


def get_first_x_start_x_stop(hist_projections, thres=5):
    i = 0
    x_start_rel = 0
    x_stop_rel = 0
    while(i < hist_projections.shape[0] and hist_projections[i] >= thres):
        i += 1
    x_start_rel = i
    x_stop_rel = i
    while(i < hist_projections.shape[0] and hist_projections[i] < thres):
        i += 1
    x_stop_rel = i
    return x_start_rel, x_stop_rel


def cutline(line_bbox, labels_check, bboxs):
    groups = []
    x_start, x_stop, y_start, y_stop = get_x_y_start_stop(line_bbox)
    sub_region_labels = labels_check[y_start:y_stop, x_start:x_stop]
    print(x_stop-x_start)
    hist_projections = np.sum(sub_region_labels, axis=0)
    cursor = 0
    x_start_next = x_start
    while(cursor < x_stop-x_start):
        x_start_rel, x_stop_rel = get_first_x_start_x_stop(
            hist_projections[cursor:])
        groups.append(get_boundingbox_tl_lr(
            (y_start, x_start_next+x_start_rel),
            (y_stop, x_start_next+x_stop_rel)))
        x_start_next = x_start_next+x_stop_rel
        cursor = cursor + x_stop_rel

    print(hist_projections.shape)
    final_groups = []
    for cutted_rect in groups:
        is_intersect = False
        x_start, x_stop, _, _ = get_x_y_start_stop(cutted_rect)
        if (x_stop - x_start) > 7:
            for bbox in bboxs:
                if check_intersect_x(bbox, cutted_rect):
                    is_intersect = True
                    break
        if is_intersect:
            final_groups.append(cutted_rect)
    return final_groups


def check_projection_cell_merge_bbox(bboxs, labels, index, edges):
    non_table_bboxs = []
    textual_bboxs = []
    labels_check = np.array((edges != 0)*1)
    print("Total number of bbox: ", len(bboxs))
    non_table_bboxs = [bbox for (i, bbox) in enumerate(bboxs) if i != index]
    print("Total number of nontable bbox: ", len(non_table_bboxs))

    non_table_bboxs, rects_types,\
        areas, width_textual_bbox,\
        height_textual_bbox = get_filtered_bboxs_tawh(non_table_bboxs, 7)

    print("Total number of not so big nontable bbox: ", len(non_table_bboxs))
    if len(areas) == 0:
        area_avg = 0
    else:
        area_avg = sum(areas)/len(areas)
    if len(height_textual_bbox) == 0:
        height_avg = 0
    else:
        height_avg = sum(height_textual_bbox)/len(height_textual_bbox)
    textual_bboxs = dummy_get_textual_bboxs(non_table_bboxs,
                                            height_textual_bbox,
                                            rects_types, area_avg,
                                            height_avg)
    print("Total number of textual_bboxs: ", len(textual_bboxs))
    lines, lines_range = check_projection_merge_line(
        textual_bboxs, rects_types)
    for i, bbox_line in enumerate(lines):
        lines[i], line_ranges = sort_bbox_left_to_right(bbox_line)

    min_rects = [get_minimum_bounding_rect(line)
                 for line in lines]
    new_min_rects = []

    for i, min_rect in enumerate(min_rects):
        new_min_rects.extend(cutline(min_rect, labels_check, lines[i]))

    """
    groups = group_bbox_non_label_overlapping(new_min_rects, labels_check)
    new_min_rects = [get_minimum_bounding_rect(group) for group in groups]
    """
    print("Number of groups: ", len(min_rects))
    return new_min_rects, textual_bboxs


def invert_color_image(array_imggray):
    return np.invert(array_imggray.copy())


def is_horizontal_line(bounding_rect):
    return get_aspect_ratio_bounding_box(bounding_rect) > 15


def is_vertical_line(bounding_rect):
    return get_aspect_ratio_bounding_box(bounding_rect) < 1.0/15.0


def find_and_keep_biggest_areas(big_areas):
    """
    Do exactly as the name
    Should be optimized to be faster
    """
    new_big_areas = []
    for big_area in big_areas:
        if len(new_big_areas) == 0:
            new_big_areas.append(big_area)
        else:
            intersect = False
            for index, new_big_area in enumerate(new_big_areas):
                intersect, swapped = check_bbox_contains_each_other(
                    new_big_area, big_area)
                if intersect:
                    if swapped:
                        new_big_areas.remove(new_big_area)
                        new_big_areas.append(big_area)
                    break
            if not intersect:
                new_big_areas.append(big_area)
    return new_big_areas


def ocropy_binarize_grayscale(image_path):
    """
    return two image
    """
    subprocess.call(["ocropus-nlbin", "-n", image_path])
    image_dir, image_name = os.path.split(image_path)
    grayscale_image_name = "".join([image_name.split(".")[0], ".nrm.png"])
    grayscale_image_path = os.path.join(image_dir, grayscale_image_name)
    bin_image_name = "".join([image_name.split(".")[0], ".bin.png"])
    bin_image_path = os.path.join(image_dir, bin_image_name)
    return cv2.imread(bin_image_path),\
        np.array(Image.open(grayscale_image_path).convert('L'))


def connected_component_analysis_sk_phase2(bbox_img, array_imggray,
                                           neighbors=8, connectivity=1,
                                           padding=0):
    labels = skimage.morphology.label(array_imggray, neighbors,
                                      background=0,
                                      connectivity=connectivity)
    areas = []
    objs = ndimage.find_objects(labels)
    bboxs = []
    for _obj in objs:
        bboxs.append(get_boundingbox_tl_lr([_obj[0].start,
                                           _obj[1].start],
                                           [_obj[0].stop,
                                            _obj[1].stop]))
        areas.append(get_area_bounding_box(bboxs[-1]))
    index = np.argmax(areas)
    array_imggray2 = np.array((labels == (index+1))*255, dtype=np.uint8)
    edges = cv2.Canny(array_imggray2, 20, 150, apertureSize=3)
    return edges, labels, bboxs, objs, index


def connected_component_analysis_sk_phase3(bbox_img, array_imggray,
                                           neighbors=8, connectivity=1,
                                           padding=0):
    labels = skimage.morphology.label(array_imggray, neighbors,
                                      background=0,
                                      connectivity=connectivity)
    areas = []
    objs = ndimage.find_objects(labels)
    for _obj in objs:
        areas.append(get_area_bounding_box(get_boundingbox_tl_lr(
            [_obj[0].start, _obj[1].start],
            [_obj[0].stop, _obj[1].stop])))
    index = np.argmax(areas)
    array_imggray2 = np.array((labels == (index+1))*255, dtype=np.uint8)
    edges = cv2.Canny(array_imggray2, 50, 150, apertureSize=3)

    lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
    if lines is not None:
        for rho, theta in lines[0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))

            cv2.line(bbox_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    # imshow(edges)
    # show()
    return edges


def get_filtered_bboxs_tawh(bboxs, thres=10):
    "type area width height"
    rects_types = []
    bounding_rects = []
    heights = []
    widths = []
    areas = []
    for new_rect in bboxs:
        is_addable, width, height = dummy_filter_bbox_height_width(new_rect,
                                                                   thres)
        if is_addable:
            bounding_rects.append(new_rect)
            widths.append(width)
            heights.append(height)
            areas.append(get_area_bounding_box(new_rect))
            if is_horizontal_line(new_rect):
                rects_types.append(1)
            else:
                rects_types.append(0)
    return bounding_rects, rects_types, areas, widths, heights


def connected_component_analysis_sk_with_labels(
        array_imggray,
        neighbors=8, connectivity=1,
        padding=0):
    """
    labels, bboxs
    """
    labels = skimage.morphology.label(array_imggray, neighbors,
                                      background=0,
                                      connectivity=connectivity)
    objs = ndimage.find_objects(labels)

    bboxs = get_objs_bboxs(objs)
    return labels, bboxs


def connected_component_analysis_sk(array_imggray, neighbors=8, connectivity=1,
                                    padding=0):
    return connected_component_analysis_sk_with_labels(
        array_imggray, neighbors, connectivity, padding)[1]


def connected_component_analysis_sk_merge(array_imggray,
                                          neighbors=8, connectivity=1,
                                          padding=0):
    labels = skimage.morphology.label(array_imggray, neighbors,
                                      background=0,
                                      connectivity=connectivity)
    bounding_rects = []
    areas = []
    rects_types = []
    # num_list = np.unique(labels)
    objs = ndimage.find_objects(labels)

    heights = []
    bboxs = get_objs_bboxs(objs)
    bounding_rects, rects_types,\
        areas, widths, heights = get_filtered_bboxs_tawh(bboxs, 10)
    text_rects = [rect for (index, rect) in enumerate(bounding_rects)
                  if dummy_hard_code_rule_height(index, bounding_rects,
                                                 heights, rects_types)]
    if len() == 0:
        return None, None, None, None, None
    avg_areas = sum(areas)/len(areas)
    big_areas = [rect for (index, rect) in enumerate(bounding_rects) if
                 dummy_hard_code_rule_big_area_height(
                     index, bounding_rects, heights,
                     rects_types, avg_areas) > 2]
    print("Number of big areas: ", len(big_areas))
    new_big_areas = find_and_keep_biggest_areas(big_areas)

    lines, lines_y_ranges = check_projection_merge_line(text_rects,
                                                        [0]*len(text_rects))
    real_lines = []
    for i, line in enumerate(lines):
        separated_lines_groups, ligroups, cross = count_merged_lines(line)
        real_lines.extend(separated_lines_groups)
    real_lines = np.array(real_lines)
    return bounding_rects, rects_types, real_lines, lines_y_ranges,\
        new_big_areas


def connected_component_analysis(array_imggray, connectivity=8, padding=0):
    # CV_16U
    output = cv2.connectedComponentsWithStats(array_imggray, connectivity, 2)
    labels = output[1]
    stats = output[2]
    # Statistic for the outputs, stats[label, COLUMN]
    # COLUMN can be cv2.CC_STAT_LEFT
    #               cv2.CC_STAT_TOP
    #               cv2.CC_STAT_WIDTH
    #               cv2.CC_STAT_HEIGHT
    #               cv2.CC_STAT_AREA
    rects = []
    areas = []
    print(labels.shape)
    print(stats.shape)
    for i, label in enumerate(np.unique(labels)):
        if label == 0 or stats[label, cv2.CC_STAT_AREA] < 30:
            continue
        rects.append(get_boundingbox_stats(stats, label, padding))
        areas.append(stats[label, cv2.CC_STAT_AREA])
    return rects, areas


def tesseract4_wrap(img, bbox, out_image_path="temp.jpg",
                    out_text_path="output_tess_last"):
    crop_bbox_to_file(img, bbox, out_image_path)
    if out_image_path is None:
        out_image_path = "temp.jpg"
    if out_text_path is None:
        out_text_path = "output_tess_last"
    while not os.path.exists(out_image_path):
        pass
    os.system("tesseract4 \"$@\" -l jpn\
              --psm 3 -c chop_enable=T\
              -c use_new_state_cost=F\
              -c segment_segcost_rating=F\
              -c enable_new_segsearch=0\
              -c language_model_ngram_on=0\
              -c textord_force_make_prop_words=F\
              -c edges_max_children_per_outline=40\
               " + out_image_path + " " + out_text_path)
    return re.sub(r'\s+', '', open(out_text_path+".txt", "r").read())


def tesseract4_wrap_all_image(out_image_path="temp.jpg",
                              out_text_path="output_tess_last"):
    os.system("tesseract4 \"$@\" -l jpn\
              --psm 3 -c chop_enable=T\
              -c use_new_state_cost=F\
              -c segment_segcost_rating=F\
              -c enable_new_segsearch=0\
              -c language_model_ngram_on=0\
              -c textord_force_make_prop_words=F\
              -c edges_max_children_per_outline=40\
               " + out_image_path + " " + out_text_path)
    return re.sub(r'\s+', '', open(out_text_path+".txt", "r").read())


def tesseract_wrap(img, bbox, out_image_path="temp.jpg",
                   out_text_path="output_tess_last"):
    crop_bbox_to_file(img, bbox, out_image_path)
    if out_image_path is None:
        out_image_path = "temp.jpg"
    if out_text_path is None:
        out_text_path = "output_tess_last"
    while not os.path.exists(out_image_path):
        pass
    os.system("tesseract \"$@\" -l jpn\
              --psm 3 -c chop_enable=T\
              -c use_new_state_cost=F\
              -c segment_segcost_rating=F\
              -c enable_new_segsearch=0\
              -c language_model_ngram_on=0\
              -c textord_force_make_prop_words=F\
              -c edges_max_children_per_outline=40\
               " + out_image_path + " " + out_text_path)
    return re.sub(r'\s+', '', open(out_text_path+".txt", "r").read())


def tesseract_wrap_all_image(out_image_path="temp.jpg",
                             out_text_path="output_tess_last"):
    os.system("tesseract \"$@\" -l jpn\
              --psm 7 -c chop_enable=T\
              -c use_new_state_cost=F\
              -c segment_segcost_rating=F\
              -c enable_new_segsearch=0\
              -c language_model_ngram_on=0\
              -c textord_force_make_prop_words=F\
              -c edges_max_children_per_outline=40\
               " + out_image_path + " " + out_text_path)
    return re.sub(r'\s+', '', open(out_text_path+".txt", "r").read())


def find_first_point_clockwise(tmp_img, b, c):
    '''
    2 3 4               6 7 8
    -1 b0 1     +4 =>   3 4 5
    -4 -3 -2            0 1 2
    '''
    list_index_clockwise = [3, 0, 1, 6, None, 2, 7, 8, 5]
    list_index_cclockwise = [1, 2, 5, 0, None, 8, 3, 6, 7]
    list_relative_pos_clockwise = np.array([
        [-1, 0], [-1, -1], [0, -1], [-1, 1], [0, 0], [1, -1],
        [0, 1], [1, 1], [1, 0]])
    starting_pos = c - b
    starting_index = starting_pos[0] + starting_pos[1]*3
    if starting_index == 0:
        raise ValueError("b0 and c0 are not supposed to be equal!")
    starting_index = int(starting_index + 4)
    index = b+list_relative_pos_clockwise[starting_index]
    while tmp_img[index[0], index[1]] == 0:
        starting_index = list_index_clockwise[starting_index]
        index = b+list_relative_pos_clockwise[starting_index]
    b1 = b+list_relative_pos_clockwise[starting_index]
    c1 = b+list_relative_pos_clockwise[
        list_index_cclockwise[starting_index]]

    return b1, c1


def border_following_clockwise(points, y_top=False, outer_border=True):
    """sort points in clockwise order
    Note: assuming right-hand x,y axis system with x first, the points must be
    connected (in atleast 8-connection)
    input:
        points: numpy array of 2d points nx2, x first, y later
        y_top: False if y-axis points down
        outer_border: True if we want to follow outerborder, not the object
    output:
        sorted points, numpy array
    """
    if y_top:
        points[:, 1] = -points[:, 1]
    # Upper most first
    up_most = max(points[:, 1])
    left_most = min(points[:, 0])
    down_most = min(points[:, 1])
    right_most = max(points[:, 0])

    starting_point = np.array([left_most-1, down_most-1])  # for padding
    tmp_points = points - starting_point
    tmp_up_most = up_most - down_most + 1
    tmp_right_most = right_most - left_most + 1

    b0_x = min(tmp_points[tmp_points[:, 1] == tmp_up_most][:, 0])
    b0 = np.array([b0_x, tmp_up_most])
    c0 = np.array([b0_x-1, up_most])

    tmp_img = np.zeros((tmp_right_most+2, tmp_up_most+2))
    tmp_img[np.transpose(tmp_points)] = 1
    # We should have a nice set of points in the middle now
    b, c = find_first_point_clockwise(tmp_img, b0, c0)
    points_outputs = []
    if not outer_border:
        points_outputs = [b0]
        while(b[0] != b0[0] and b[1] != b0[1]):
            points_outputs.append(b)
            b, c = find_first_point_clockwise(tmp_img, b, c)
    else:
        points_outputs = [c0]
        while(c[0] != c0[0] and c[1] != c0[1]):
            points_outputs.append(c)
            b, c = find_first_point_clockwise(tmp_img, b, c)

    points_outputs = np.array(points_outputs)
    points_outputs += starting_point
    if y_top:
        points_outputs[:, 1] = -points_outputs[:, 1]
    return points_outputs
