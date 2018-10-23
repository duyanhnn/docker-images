import numpy as np
import cv2
from ..bboxs_tools.bbox_operations import (
    get_area_bounding_box,
    get_width_height_bounding_box,
    get_boundingbox_tl_lr,
    )


def dummy_hard_code_rule_big_area(index, rects, rects_heights,
                                  rects_types, areas_avg=None):
    if areas_avg is None:
        areas = [get_area_bounding_box(rect) for (i, rect) in enumerate(rects)
                 if rects_types[i] == 0]
        areas_avg = sum(areas)/len(areas)
    rect_area = get_area_bounding_box(rects[index])
    if rect_area > 6*areas_avg:
        if rects_types[index] == 1:     # Is this classified as a line?
            rects_types[index] = 3       # Horizontal Box
        elif rects_types[index] == 2:   # Is this classified as a vertical line
            rects_types[index] = 4
        else:                           # No aspect ratio
            rects_types[index] = 5       # Big area, unknown

    return rects_types[index]


def dummy_hard_code_rule_big_area_height(index, rects, rects_heights,
                                         rects_types, areas_avg=None,
                                         heights_avg=None):
    if areas_avg is None:
        areas = [get_area_bounding_box(rect) for (i, rect) in enumerate(rects)
                 if rects_types[i] == 0]
        areas_avg = sum(areas)/len(areas)

    if heights_avg is None:
        text_only_heights = [height for (i, height) in enumerate(rects_heights)
                             if rects_types[i] == 0]
        heights_avg = sum(text_only_heights)/len(text_only_heights)

    rect_area = get_area_bounding_box(rects[index])
    if rect_area > 6*areas_avg:
        if rects_types[index] == 0 and rects_heights[index] > 2*heights_avg:
            # Is this classified as a line?
            rects_types[index] = 3       # Horizontal Box
        elif rects_types[index] == 2:   # Is this classified as a vertical line
            rects_types[index] = 4
        else:                           # No aspect ratio
            rects_types[index] = 5       # Big area, unknown

    return rects_types[index]


def dummy_hard_code_rule_height(index, rects, rects_heights, rects_types):
    if rects_types[index] != 0:
        return False
    text_only_heights = [height for (i, height) in enumerate(rects_heights)
                         if rects_types[i] == 0]
    avg_heights = sum(text_only_heights)/len(text_only_heights)
    if rects_heights[index] >= 1.5*avg_heights:
        return False
    elif rects_heights[index] <= 0.66*avg_heights:
        return False
    else:
        return True


def dummy_filter_bbox_height_width(bbox, thres=10):
    width, height = get_width_height_bounding_box(bbox)
    if height > thres or width > thres:
        return True, width, height
    return False, None, None


def dummy_get_textual_bboxs(bboxs, heights, rects_types, avg_area, avg_height):
    textual_bboxs = []
    for i, bbox in enumerate(bboxs):
        if dummy_hard_code_rule_big_area_height(i, bboxs, heights, rects_types,
                                                avg_area, avg_height) > 2:
            # Perhaps more appropiating process can be done
            continue
        if rects_types[i] == 1:
            continue
        textual_bboxs.append(bbox)
    return textual_bboxs


def dummy_find_blackbox_table_2(im):
    lower_res = im
    hsv = cv2.cvtColor(lower_res, cv2.COLOR_BGR2HSV)
    if im.shape[0] < 7 or im.shape[1] < 7:
        return []
    blur = cv2.GaussianBlur(hsv, (7, 7), 0)
    HSVLOW = np.array([0, 0, 0])
    HSVHIGH = np.array([179, 40, 255])
    # apply the range on a mask
    mask = cv2.inRange(blur, HSVLOW, HSVHIGH)
    mask_inv = cv2.bitwise_not(mask)
    mask_inv = cv2.medianBlur(mask_inv, 3)
    d = cv2.cvtColor(mask_inv, cv2.COLOR_GRAY2BGR)
    _, contours, hierarchy = cv2.findContours(mask_inv, 1, 2)
    list_rects = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if (area < 200):
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        list_rects.append(get_boundingbox_tl_lr((y, x), (y+h, x+w)))
        cv2.rectangle(d, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return list_rects
