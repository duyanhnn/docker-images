from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import cv2
import numpy as np


def get_x_y_start_stop(bounding_box):
    """
    return x_start, x_end, y_start, y_end, image coordinate of a bbox
    """
    x_start = bounding_box[0][0]
    y_start = bounding_box[3][1]
    x_stop = bounding_box[1][0]
    y_stop = bounding_box[0][1]
    return x_start, x_stop, y_start, y_stop


def check_intersect_interval(interval_a, interval_b):
    start_a = interval_a[0]
    stop_a = interval_a[1]
    start_b = interval_b[0]
    stop_b = interval_b[1]
    if start_a >= start_b:
        start_a, start_b = start_b, start_a
        stop_a, stop_b = stop_b, stop_a
    if start_b <= stop_a:
        return True, min(start_a, start_b), max(stop_a, stop_b)
    return False, None, None


def check_intersect_x(rect_a, rect_b):
    """Check if two bbox intersected - horizontal"""
    x_start_a, x_stop_a, _, _ = get_x_y_start_stop(rect_a)
    x_start_b, x_stop_b, _, _ = get_x_y_start_stop(rect_b)
    return check_intersect_interval((x_start_a, x_stop_a),
                                    (x_start_b, x_stop_b))[0]


def check_intersect_y(rect_a, rect_b):
    """Check if two bbox intersected - vertical"""
    _, _, y_start_a, y_stop_a = get_x_y_start_stop(rect_a)
    _, _, y_start_b, y_stop_b = get_x_y_start_stop(rect_b)
    return check_intersect_interval((y_start_a, y_stop_a),
                                    (y_start_b, y_stop_b))[0]


def check_bbox_intersect(bbox_a, bbox_b):
    return check_intersect_x(bbox_a, bbox_b) and\
        check_intersect_y(bbox_a, bbox_b)


def merge_intersected_bboxs(bboxs):
    i = 0
    while i < len(bboxs):
        j = i+1
        while j < len(bboxs):
            if check_bbox_intersect(bboxs[i], bboxs[j]):
                bboxs[i] = get_minimum_bounding_rect(
                    np.array([bboxs[i], bboxs[j]]))
                bboxs.pop(j)
    return bboxs


def get_width_height_bounding_box(bounding_rect):
    height = abs(bounding_rect[0][1] - bounding_rect[3][1])
    width = abs(bounding_rect[0][0] - bounding_rect[1][0])
    return width, height


def get_width_height_bounding_boxs(bounding_rect_list):
    widths = []
    heights = []
    for bounding_rect in bounding_rect_list:
        width, height = get_width_height_bounding_box(bounding_rect)
        widths.append(width)
        heights.append(height)

    return widths, heights


def get_aspect_ratio_bounding_box(bounding_rect):
    width, height = get_width_height_bounding_box(bounding_rect)
    if height == 0:
        return 100
    return width/height


def get_area_bounding_box(bounding_rect):
    width, height = get_width_height_bounding_box(bounding_rect)
    return width*height


def get_boundingbox_tl_lr(top_left, lower_right, padding=0):
    # input: y,x
    # output: x,y
    rect = []
    top_left = [top_left[1] - padding,
                top_left[0] + padding]
    lower_right = [lower_right[1] + padding,
                   lower_right[0] - padding]
    rect = [[top_left[0], lower_right[1]],
            lower_right,
            [lower_right[0], top_left[1]],
            top_left]
    return np.array(rect)


def get_boundingbox_stats(stats, label, padding=0):
    rect = []
    top_left = [stats[label, cv2.CC_STAT_LEFT] - padding,
                stats[label, cv2.CC_STAT_TOP] + padding]
    lower_right = [top_left[0] + stats[label, cv2.CC_STAT_WIDTH] + 2*padding,
                   top_left[1] + stats[label, cv2.CC_STAT_HEIGHT] + 2*padding]
    rect = [[top_left[0], lower_right[1]],
            lower_right,
            [lower_right[0], top_left[1]],
            top_left]
    return np.array(rect)


def cal_adj_y_matrix_rects(rects, same_val=0):
    """
    Build adjacent matrix on bounding box intersection: two boxes have
    connection when their projection on y axis intersect
    """
    adj_mat = np.zeros((len(rects), len(rects)))
    for i, rect in enumerate(rects):
        adj_mat[i, i] = same_val
        y_start = rect[3][1]
        y_stop = rect[0][1]
        for j in range(i+1, len(rects)):
            if check_intersect_interval(
                    (y_start, y_stop), (rects[j][3][1], rects[j][0][1]))[0]:
                adj_mat[i, j] = 1
                adj_mat[j, i] = 1
    # print(adj_mat)
    return adj_mat


def cal_sub_adj_matrix(parent_adj_mat, sorted_indices):
    """
    """
    # TESTED:
    # Sort indices from small to large
    # new indices in new sub adj_matrix
    new_adj_mats = np.zeros([len(sorted_indices), len(sorted_indices)])
    for i, current_index in enumerate(sorted_indices):
        new_adj_mats[i, :] = parent_adj_mat[current_index, sorted_indices]
    return new_adj_mats


def cal_total_intersect_groups(adj_mat, line_index_groups, current_index):
    """
    calculate the intersect with available line_groups
    """
    total_intersects = np.zeros(len(line_index_groups))
    # print("line_index_groups: ", line_index_groups)
    # print(range(len(line_index_groups)))
    for j in range(len(line_index_groups)):
        total_intersects[j] = adj_mat[current_index,
                                      line_index_groups[j][0]]
    return total_intersects


def count_merged_lines_first_pass(in_line_rects, adj_mat=None):
    """
    First pass, return non-recursive calculated groups
    """
    # print("line len: ", len(in_line_rects))
    # print("Input first pass: ", in_line_rects)
    if adj_mat is None:
        adj_mat = cal_adj_y_matrix_rects(in_line_rects, 1)
    # print("adj_mat: ", adj_mat)
    num_sum_total = adj_mat.shape[0]
    num_intersects = np.sum(adj_mat, axis=1)
    # print("Num_intersects: ", num_intersects)
    traversed_rects = []
    line_groups = []
    line_index_groups = []
    # each elem is a tuple ( index, [intersect_group1 ,intersect_group2, ...])
    cross_intersecting_rects = []
    cross_intersecting_rects_first_pass = []
    # indices that belongs to all groups
    for i, sum_intersects in enumerate(num_intersects):
        current_rect = in_line_rects[i]
        # print("num_sum_total: ", num_sum_total,
        #      "sum_intersects: ", sum_intersects)
        if num_sum_total > sum_intersects:
            if i not in traversed_rects:
                traversed_rects.append(i)
            else:
                continue
            if len(line_groups) == 0:
                line_groups.append([current_rect])
                line_index_groups.append([i])
                # print("adj_mat, lines i: ", adj_mat[i, :])
                index_opposites = np.where(adj_mat[i, :] == 0)[0]
                # print("index opposites: ", index_opposites)
                index_opposite = index_opposites[0]
                # print("first index opposite: ", index_opposite)
                # The first opposite index
                line_index_groups.append([index_opposite])
                line_groups.append([in_line_rects[index_opposite]])
                traversed_rects.append(index_opposite)
                continue
            else:
                total_intersects = cal_total_intersect_groups(
                    adj_mat, line_index_groups, i)
                # print("Total intersects: ", total_intersects)
                total_groups_belong = sum(total_intersects)
                if total_groups_belong == 1:
                    group_belongs = np.where(total_intersects == 1)
                    group_belong = group_belongs[0][0]
                    # print("group_belong: ", group_belong)
                    line_groups[group_belong].append(current_rect)
                    line_index_groups[group_belong].append(i)
                elif total_groups_belong > 1:
                    group_belongs = np.where(total_intersects == 1)[0]
                    # print("group belongs: ", group_belongs)
                    cross_intersecting_rects.append((i, group_belongs))
                else:
                    line_groups.append([current_rect])
                    line_index_groups.append([i])
        else:
            cross_intersecting_rects_first_pass.append(i)
    if len(line_index_groups) > 0:
        for i in cross_intersecting_rects_first_pass:
            cross_intersecting_rects.append((i, range(len(line_index_groups))))
    # print(len(line_groups))
    return line_groups, line_index_groups, cross_intersecting_rects, adj_mat


def count_merged_lines_recursive_step(line_groups_fp, line_index_groups_fp,
                                      cross_intersecting_rects_fp, adj_mat):
    # recursive step: keep splitting groups until only one line groups remained
    # and returned
    line_groups = []
    line_index_groups = []
    cross_intersecting_rects = []
    remapping_cross_intersecting_rects = []
    for i, line_group in enumerate(line_groups_fp):
        # Build adjacent matrix for each group
        adj_matrix_line_group = cal_sub_adj_matrix(adj_mat,
                                                   line_index_groups_fp[i])
        sub_line_groups, sub_line_index_groups,\
            sub_cross_intersecting_rects = count_merged_lines(
                line_group, adj_matrix_line_group)
        # Map these into old indices
        new_sub_line_index_groups = []
        for sub_line_index_group in sub_line_index_groups:
            new_sub_line_index_groups.append([])
            for offseted_index in sub_line_index_group:
                real_index = line_index_groups_fp[i][offseted_index]
                new_sub_line_index_groups[-1].append(real_index)
        sub_line_index_groups = new_sub_line_index_groups
        # Add cross_intersecting_rects to this
        for j, cross_intersecting_rect in enumerate(
                sub_cross_intersecting_rects):
            group_belonging = [index + len(line_groups)
                               for index in cross_intersecting_rect[1]]
            group_intersecting_index = line_index_groups_fp[i][
                cross_intersecting_rect[0]]
            cross_intersecting_rects.append(
                (group_intersecting_index, group_belonging))
        remapping_cross_intersecting_rects.append(
            [index + len(line_index_groups)
             for index in range(len(sub_line_index_groups))])
        line_groups.extend(sub_line_groups)
        line_index_groups.extend(sub_line_index_groups)
    # print("Mapping group: ", remapping_cross_intersecting_rects)
    # print("cross_intersecting_rects after fp: ", cross_intersecting_rects_fp)
    for i, cross_intersecting_rect in enumerate(cross_intersecting_rects_fp):
        new_cross_intersecting_rect = (cross_intersecting_rect[0], [])
        # print(cross_intersecting_rect)
        for j in cross_intersecting_rect[1]:
            new_cross_intersecting_rect[1].extend(
                remapping_cross_intersecting_rects[j])
        cross_intersecting_rects.append(new_cross_intersecting_rect)
    return line_groups, line_index_groups, cross_intersecting_rects


def count_merged_lines(in_line_rects, adj_mat=None):
    """
    Counting number of actual lines in merged line rects
    Return:
        line_groups: groups of rects in the same line, no merge
        line_index_groups: groups of indices of rects in the same line, no merge
        cross_intersecting_rects: groups of indices of rects in multiple line
    """
    # print(in_line_rects)
    line_groups_fp, line_index_groups_fp,\
        cross_intersecting_rects_fp, adj_mat = count_merged_lines_first_pass(
            in_line_rects,
            adj_mat)
    # print("line group fp: ", line_groups_fp)
    if len(line_groups_fp) == 0:  # Not separable
        return [in_line_rects], [range(len(in_line_rects))], []
    if len(line_groups_fp) == 2 and len(in_line_rects) == 2:
        return line_groups_fp, line_index_groups_fp, cross_intersecting_rects_fp
    return count_merged_lines_recursive_step(
        line_groups_fp, line_index_groups_fp,
        cross_intersecting_rects_fp, adj_mat)


def check_bbox_contains_each_other(bounding_box_a, bounding_box_b):
    """ do exactly as the name said
    note: exact overlap is counted as contained
    return: True/False & swapped
    """
    swapped = False
    if bounding_box_a[3][1] > bounding_box_b[3][1]:
        bounding_box_a, bounding_box_b = bounding_box_b, bounding_box_a
        swapped = True
    x_rect_start_a, x_rect_stop_a,\
        y_rect_start_a, y_rect_stop_a = get_x_y_start_stop(bounding_box_a)
    x_rect_start_b, x_rect_stop_b,\
        y_rect_start_b, y_rect_stop_b = get_x_y_start_stop(bounding_box_b)

    if (
        y_rect_start_a <= y_rect_start_b and
        y_rect_stop_a >= y_rect_stop_b and
        x_rect_start_a <= x_rect_start_b and
        x_rect_stop_a >= x_rect_stop_b
    ):
        return True, swapped
    return False, swapped


def get_areas_avg(rects):
    areas = [get_area_bounding_box(rect) for rect in rects]
    return sum(areas)/len(areas)


def crop_bbox(img, bbox):
    """Return a smaller region of image that the bounding box occupy
    """
    x_start, x_stop, y_start, y_stop = get_x_y_start_stop(bbox)
    return img[y_start:y_stop, x_start:x_stop]


def crop_bbox_to_file(img, bbox, output_filepath):
    cv2.imwrite(output_filepath, crop_bbox(img, bbox))


def check_empty_interval(labels_check,
                         x_start_interval, x_stop_interval,
                         y_start_interval, y_stop_interval):
    return np.sum(labels_check[
                    y_start_interval:y_stop_interval,
                    x_start_interval:x_stop_interval]) == 0


def sort_bbox_left_to_right(list_of_bboxs):
    x_starts = np.array(list_of_bboxs)[:, 0, 0]
    x_stops = np.array(list_of_bboxs)[:, 1, 0]
    x_mediums = (x_starts + x_stops)/2
    sorted_index = [i[0] for i in sorted(enumerate(x_mediums),
                                         key=lambda x:x[1])]
    list_of_bboxs = [list_of_bboxs[index] for index in sorted_index]
    return list_of_bboxs, sorted_index


def sort_bbox_top_down(list_of_bboxs):
    y_starts = np.array(list_of_bboxs)[:, 3, 1]
    y_stops = np.array(list_of_bboxs)[:, 0, 1]
    y_mediums = (y_starts + y_stops)/2
    sorted_index = [i[0] for i in sorted(enumerate(y_mediums),
                                         key=lambda x:x[1])]
    list_of_bboxs = [list_of_bboxs[index] for index in sorted_index]
    return list_of_bboxs, sorted_index


def sort_bbox_bottom_up(list_of_bboxs):
    top_down, sorted_index = sort_bbox_top_down(list_of_bboxs)
    return top_down[::-1], sorted_index[::-1]


def get_objs_bboxs(objs):
    rects = [get_boundingbox_tl_lr([_obj[0].start, _obj[1].start],
                                   [_obj[0].stop, _obj[1].stop])
             for _obj in objs]
    return rects


def get_minimum_bounding_rect(rects):
    """
    Get the minimum bbox that can contains all the bboxs
    """
    # TESTED! NO need to reprint
    rects = np.array(rects)
    top_left_rects = rects[:, 3, :]
    min_xs, min_ys = top_left_rects[:, 0], top_left_rects[:, 1]
    bottom_right_rects = rects[:, 1, :]
    max_xs, max_ys = bottom_right_rects[:, 0], bottom_right_rects[:, 1]
    min_x, max_x, min_y, max_y = min(min_xs), max(max_xs), min(min_ys), max(
        max_ys)
    return get_boundingbox_tl_lr((min_y, min_x), (max_y, max_x))


def get_all_bbox_not_contained_by_bbox_b(bboxs, bbox_bs):
    new_bboxs = []
    for bbox in bboxs:
        is_intersect = False
        for black_box in bbox_bs:
            if check_bbox_contains_each_other(bbox, black_box)[0]:
                is_intersect = True
                break
        if not is_intersect:
            new_bboxs.append(bbox)
    return new_bboxs


def map_bbox_to_parent_bbox(bbox, parent_bbox):
    x_start_c, x_stop_c, y_start_c, y_stop_c = get_x_y_start_stop(bbox)
    x_start_p, _, y_start_p, _ = get_x_y_start_stop(parent_bbox)
    x_start_c += x_start_p
    x_stop_c += x_start_p
    y_start_c += y_start_p
    y_stop_c += y_start_p
    return get_boundingbox_tl_lr((y_start_c, x_start_c), (y_stop_c, x_stop_c))


def cut_bbox_take_right(bbox, ratio):
    x_start, x_stop, y_start, y_stop = get_x_y_start_stop(bbox)
    if x_stop-x_start > 0:
        x_start = int(x_start + (x_stop-x_start)*ratio)
    return get_boundingbox_tl_lr((y_start, x_start), (y_stop, x_stop))


def cut_bbox_take_left(bbox, ratio):
    x_start, x_stop, y_start, y_stop = get_x_y_start_stop(bbox)
    if x_stop-x_start > 0:
        x_stop = int(x_start + (x_stop-x_start)*ratio)
    return get_boundingbox_tl_lr((y_start, x_start), (y_stop, x_stop))


def cut_bbox_take_range_x(bbox, ratio_start_x, ratio_stop_x):
    x_start, x_stop, y_start, y_stop = get_x_y_start_stop(bbox)
    if x_stop-x_start > 0:
        x_start, x_stop = int(x_start + (x_stop-x_start)*ratio_start_x),\
            int(x_start + (x_stop-x_start)*ratio_stop_x)
    return get_boundingbox_tl_lr((y_start, x_start), (y_stop, x_stop))


def cut_bbox_take_range_y(bbox, ratio_start_y, ratio_stop_y):
    x_start, x_stop, y_start, y_stop = get_x_y_start_stop(bbox)
    if y_stop-y_start > 0:
        y_start, y_stop = int(y_start + (y_stop-y_start)*ratio_start_y),\
            int(y_start + (y_stop-y_start)*ratio_stop_y)
    return get_boundingbox_tl_lr((y_start, x_start), (y_stop, x_stop))


def cut_bbox_take_range_x_y(bbox, ratio_start_x, ratio_stop_x,
                            ratio_start_y, ratio_stop_y):
    return cut_bbox_take_range_y(
        cut_bbox_take_range_x(bbox, ratio_start_x, ratio_stop_x),
        ratio_start_y, ratio_stop_y)


def map_ratio_to_parent_ref_points(ref_pt, parent_ref_pt):
    """
    ref_pt: [(x_start, y_start), (x_end, y_end)]
    """
    sub_table_x_range = [
        (ref_pt[0][0] - parent_ref_pt[0][0]
         )/(parent_ref_pt[1][0] - parent_ref_pt[0][0]),
        (ref_pt[1][0] - parent_ref_pt[0][0]
         )/(parent_ref_pt[1][0] - parent_ref_pt[0][0])
    ]
    sub_table_y_range = [
        (ref_pt[0][1] - parent_ref_pt[0][1]
         )/(parent_ref_pt[1][1] - parent_ref_pt[0][1]),
        (ref_pt[1][1] - parent_ref_pt[0][1]
         )/(parent_ref_pt[1][1] - parent_ref_pt[0][1])
    ]
    return sub_table_x_range, sub_table_y_range


def scale_bbox_ref_point(ref_pt, ratio):
    """
    ref_pt: x_start, y_start, x_end, y_end
    """
    return get_boundingbox_tl_lr(
        (int(ref_pt[0][1]*ratio), int(ref_pt[0][0]*ratio)),
        (int(ref_pt[1][1]*ratio), int(ref_pt[1][0]*ratio))
        )


def ref_point_to_bbox(ref_pt):
    """
    ref_pt: x_start, y_start, x_end, y_end
    """
    return get_boundingbox_tl_lr(
        (ref_pt[0][1], ref_pt[0][0]),
        (ref_pt[1][1], ref_pt[1][0])
        )


def fill_box_bin_img(binimg, bbox, val):
    binimg = binimg.copy()
    x_start, x_stop, y_start, y_stop = get_x_y_start_stop(bbox)
    binimg[y_start:y_stop, x_start:x_stop] = val
    return binimg


def get_bbox_x_y_w_h(x, y, w, h):
    return get_boundingbox_tl_lr([y, x], [y+h, x+w])


get_bbox_cv = get_bbox_x_y_w_h


def get_x_y_start_stop_contour(contour):
    """Get x0, y0, x1, y1 of contour(internal format)
    Args:
        contour: a single opencv contour
    """
    contour = contour.reshape((contour.shape[0], 2))
    contour[:, [0, 1]] = contour[:, [1, 0]]  # to y, x
    x0 = np.min(contour[:, 1])
    y0 = np.min(contour[:, 0])
    x1 = np.max(contour[:, 1])
    y1 = np.max(contour[:, 0])
    return x0, y0, x1, y1


def get_bbox_cv_contour(contour):
    """Get bounding box (internal format)
    Args:
        contour: a single opencv contour
    """
    x0, y0, x1, y1 = get_x_y_start_stop_contour(contour)
    return get_bbox_x_y_w_h(x0, y0, x1-x0+1, y1-y0+1)
