from skimage.morphology import skeletonize
from scipy.ndimage.morphology import binary_dilation, binary_closing
import cv2
import numpy as np
from skimage.filters import threshold_sauvola
from .morph import width, height, ycenter, xcenter, connected_components, calculate_cell_cut, \
    r_dilation, scale_rect, r_opening, get_rects_by_contour, filter_overlap_boxes_bigger, getContours\
    , scale_pts, sort_contours_hor
from .util import resize_to_prefered_height, resize_to_prefered_width
from scipy.spatial.distance import cdist, euclidean
import os
import json

def skelet(img, thres = 150, expand = False, expand_horizontal = True, iter=1):
    img = img > thres

    img = skeletonize(img)
    # img = binary_erosion(img, iterations=1)
    img = binary_dilation(img, iterations=iter)

    if expand:
        print('Expanding mask')
        pad = 5
        kernel_shape = (1, pad) if expand_horizontal else (pad, 1)
        kernel = np.ones(kernel_shape, dtype='uint8')
        img = binary_dilation(img, kernel, iterations=1)

    return img

def find_lower_line_bound(region, min_line_thres = 0):
    h, w = region.shape
    if min(w,h) <= 0:
        return 0
    if min_line_thres == 0:
        min_line_thres = w * 0.3
    if min_line_thres > w * 0.7:
        min_line_thres = w * 0.7
    _, objects = connected_components(region)
    objects = [o for o in objects if width(o) > min_line_thres]
    objects.sort(key=lambda o: ycenter(o))
    if len(objects) > 0:
        return h - objects[-1][0].start
    else:
        return 0


def estimate_box_from_area_mask(region, baseline, baseline_full, force_straight_box = False):
    h, w = region.shape
    if min(w, h) <= 0:
        return 0, None

    expand_down_thres = 5
    baseline_expand_down = r_dilation(baseline, (2 * expand_down_thres + 1, 1), origin=(expand_down_thres,0))
    baseline = r_dilation(baseline, (3,1))
    baseline_full = r_dilation(baseline_full, (3, 1))
    bin_with_baseline = region & (1 - baseline_expand_down) #((region | baseline) * 255).astype('uint8')  # region | baseline
    open_thres = int(w * 0.2)
    bin_with_baseline = r_opening(bin_with_baseline, (1, open_thres))
    baseline_full = r_dilation(baseline_full, (1,15))
    bin_with_baseline = bin_with_baseline & (1 - baseline_full)
    bin_with_baseline = ((bin_with_baseline | baseline) * 255).astype('uint8')

    # cv2.imshow('lol', bin_with_baseline)
    # cv2.waitKey(0)
    contours = getContours(bin_with_baseline)

    width_thres = w * 0.8 if w > 8 else 3
    height_thres = 4

    cnt_boxes = []
    for cnt in contours:
        bounding_rect = cv2.boundingRect(cnt)
        rect = cv2.minAreaRect(cnt)
        (cx, cy), (w, h), angle = rect
        w, h = max(w,h), max(min(w, h), 1)
        if force_straight_box or not (h > 4 and ((1.0 * w / h > 4 and abs(angle + 90) > 0.1 and abs(angle) > 0.1))):
            x, y, w, h = bounding_rect
            box = np.array([[x,y], [x + w, y], [x + w, y + h], [x, y + h]], dtype='int64')
        else:
            box = cv2.boxPoints(rect)
            box = np.int0(box)
        cnt_boxes.append(((cx, cy), (w, h), box, cnt, bounding_rect))

    obj_y = [c[0][1] if max(c[1]) > width_thres else 0 for c in cnt_boxes]
    objects_index = np.argsort(obj_y)

    if len(objects_index) > 0:
        # object index of our line to create the mask
        i = objects_index[-1]
        line_object = cnt_boxes[i]

        mask = np.zeros(region.shape, dtype='uint8')
        # cv2.drawContours(mask, [line_object[3]], -1, (255), cv2.FILLED)
        x, y, w, h = line_object[4]
        bounding_rect = [x, y, x + w, y + h]

        return bounding_rect, line_object[2], mask
    else:
        return 0, None, None


def estimate_height_from_area_mask(region, baseline):
    h, w = region.shape
    if min(w, h) <= 0:
        return 0, None

    bin_with_baseline = region | baseline
    labels, objects = connected_components(bin_with_baseline)
    width_thres = w * 0.8 if w > 8 else 3
    obj_y = [ycenter(o) if width(o) > width_thres else 0 for o in objects]
    objects_index = np.argsort(obj_y)

    if len(objects_index) > 0:
        # object index of our line to create the mask
        i = objects_index[-1]
        line_object = objects[i]

        median_h = height(line_object)
        mask = (labels == i + 1)

        return median_h, mask
    else:
        return 0, None


def esimate_height_from_binary(region, baseline, merge_range = 0):
    h, w = region.shape
    if min(w,h) <= 0:
        return 0, None

    bin = region <= threshold_sauvola(region, window_size=15, k=0.1)
    # if merge_range > 0:
    #     bin[-merge_range:,:] = r_closing(bin[-merge_range:,:], (3,1))

    # dilation to merge small strokes
    # bin = r_closing(bin, (3, 1))

    # first connected component to get line mask
    bin_with_baseline = bin | baseline
    labels, objects = connected_components(bin_with_baseline)
    width_thres = w * 0.8 if w > 8 else 3
    obj_y = [ycenter(o) if width(o) > width_thres else 0 for o in objects]
    objects_index = np.argsort(obj_y)

    if len(objects_index) > 0:
        # object index of our line to create the mask
        i = objects_index[-1]
        line_object = objects[i]

        mask = (labels == i + 1)
        # mask = r_closing(mask, (1, 20))
        # cv2.imshow('lol', (mask * 255).astype('uint8'))
        # cv2.waitKey(1000)
        #mask = mask & bin

        # connected component to get median line height
        # _, objects = connected_components(mask)
        # obj_h = [height(o) for o in objects if height(o) > 3]
        # if len(obj_h) == 0:
        #     median_h = 0
        # else:
        #     if len(obj_h) > 2:
        #         median_h = np.percentile(obj_h, 70)
        #     else:
        #         median_h = min(obj_h)
        #         if median_h < 5:
        #             median_h = max(obj_h)

        median_h = height(line_object)

        # mask = np.zeros(mask.shape, dtype='B')

        # return height(line_object), mask

        # fill character box in the line mask
        char_count = 0
        for c in objects:
            if height(c) > median_h * 1.5 or width(c) > median_h * 2:
                continue
            char_count += 1
            x1, y1, x2, y2 = c[1].start, c[0].start, c[1].stop, c[0].stop
            char_w = x2 - x1

            # if y2 - y1 < char_w * 0.25 and h - y1 > median_h * 0.2:
            #     #y1 = int(max(0, y1 - char_w * 0.25))
            #     #y2 = int(y2 + char_w * 0.25)
            #     pass
            # else:
            #     if y2 - y1 > median_char_h * 1.2:
            #         y1 = int(y1 + char_w * 0.2)
            #         y2 = int(max(0, y2 - char_w * 0.2))

            # mask[y1:y2, x1:x2].fill(1)

        if merge_range > 3 and w > 4:
            ratio_h = 1.0 * median_h / w
            if ratio_h > 2 or (merge_range > 7 and (1.0 * median_h / merge_range > 1.5 or (median_h > 30 and median_h > merge_range))) or (ratio_h < 1 and char_count == 1 and median_h < merge_range):
                median_h = merge_range

        nice_disp = int(median_h * 0.35)  #int(median_h * 0.3)
        # upper_baseline = np.roll(baseline, -nice_disp, 0)
        # baseline = upper_baseline | baseline
        # baseline = r_closing(baseline, (nice_disp -2, 1))
        baseline = r_dilation(baseline, (2 * nice_disp + 1, 1), origin=(-nice_disp,0))
        mask = baseline #mask | baseline
        # mask[- 4 * nice_disp :-nice_disp,:].fill(1)


        # mask = r_closing(mask, (5, 50))

        return median_h, mask #h - objects[-1][0].start
    else:
        return 0, None

def estimate_height_from_side_sep(region):

    h, w = region.shape
    if min(w,h) <= 0:
        return 0
    MIN_SEP_THRES = 1
    _, objects = connected_components(region)
    objects = sorted(objects, key=lambda o: ycenter(o))
    if len(objects) > 0:
        return height(objects[-1])
    else:
        return 0

    # h, w = region.shape
    # MIN_SEP_THRES = 0
    # proj_y = np.sum(region, axis=1)
    # sep_y = np.sort([j[0] for j in np.argwhere(proj_y <= MIN_SEP_THRES)])
    # if len(sep_y) > 0:
    #     return h - sep_y[-1]
    # else:
    #     return 0

def safe_bound(x):
    return x if x > 0 else 0

def line_segmentation_old_bin(img, gt_base, gt_sep, show_im, gt_upper = None):

    labels, objects = connected_components(gt_base)
    LOOKUP_RANGE_UP = 80
    LOOKUP_RANGE_SIDE = 8
    MIN_LINE_THRES = 10

    line_boxes = []

    for i, o in enumerate(objects):
        if height(o) < 20: #and width(o) > 5:
            x1, x2 = o[1].start, o[1].stop
            y1, y2 = o[0].start, o[0].stop

            left_side_sep_region = gt_sep[safe_bound(y1 - LOOKUP_RANGE_UP):y2 + 1, safe_bound(x1 - LOOKUP_RANGE_SIDE):x1 + 2]
            right_side_sep_region = gt_sep[safe_bound(y1 - LOOKUP_RANGE_UP):y2 + 1, safe_bound(x2 - 2):x2 + LOOKUP_RANGE_SIDE]
            # import matplotlib.pyplot as plt
            #
            roi = (slice(safe_bound(y1 - LOOKUP_RANGE_UP), y2 + 1), slice(x1, x2))
            roi_y1 = (slice(safe_bound(y1 - LOOKUP_RANGE_UP), y1 - 1), slice(x1, x2))
            region = img[roi]
            baseline_mask = labels[roi] == i + 1

            distance_to_nearest_baseline = find_lower_line_bound(gt_base[roi_y1], min_line_thres=10)
            left_h = estimate_height_from_side_sep(left_side_sep_region)
            right_h = estimate_height_from_side_sep(right_side_sep_region)

            merge_range = min(left_h, right_h) if min(left_h, right_h) > 0 else max(left_h, right_h)

            distance_to_nearest_upper_line, line_mask = esimate_height_from_binary(region, baseline_mask, merge_range=merge_range)

            # distance_to_nearest_upper_line = find_lower_line_bound(gt_upper[roi_y1], min_line_thres=1)
            # print(distance_to_nearest_upper_line)
            # line_mask = np.zeros(gt_upper[roi_y1].shape, dtype='uint8')

            # height_from_sep = min(left_h, right_h) if min(left_h, right_h) > 4 else max(left_h, right_h)

            # if height_from_sep > 6 and 1.0 * distance_to_nearest_upper_line / height_from_sep > 2:
            #     distance_to_nearest_upper_line = height_from_sep

            # print('h', distance_to_nearest_upper_line)

            # fig, ax = plt.subplots(1, 3, figsize=(10, 10),
            #                        subplot_kw={'adjustable': 'box-forced'})
            #
            # ax[0].imshow(bin)
            # ax[1].imshow(bin)
            # ax[2].imshow(show_im[roi])
            # plt.show()

            # distance_to_nearest_upper_line = find_lower_line_bound(gt1[roi])

            if distance_to_nearest_baseline > 6:
                line_h = min(distance_to_nearest_baseline, distance_to_nearest_upper_line) + 3

            else:
                line_h = distance_to_nearest_upper_line + 2

            line_h = int(line_h)
            line_mask_h = line_mask.shape[0]
            mask = line_mask[safe_bound(line_mask_h - line_h - 1):, :]
            # region = show_im[y2 - mask.shape[0] + 1:y2 + 1, x1:x2]
            # region[line_mask[safe_bound(line_mask.shape[0]-line_h-1):, :] > 0] = [0, 0, 255]

            if line_h > 6:
                line_boxes.append(([x1, y2 - mask.shape[0] + 1, x2, y2 + 1], mask))

            # distance_to_nearest_line = find_lower_line_bound(upper_line_lookup_region, MIN_LINE_THRES)
            # height_left = estimate_height_from_side_sep(left_side_sep_region)
            # right_left = estimate_height_from_side_sep(right_side_sep_region)

    return line_boxes


def threshold_and_upscale_map(img_shape, gt, skeletonize=False, threshold = 150, expand = False):
    h, w = img_shape[:2]
    gt = cv2.resize(gt, (w, h))
    if skeletonize:
        gt = skelet(gt, expand=expand)
    else:
        gt = gt > threshold

    return gt

def line_segmentation_area_with_rectification(img, gt_baseline, gt_sep, gt_area):
    gt_baseline = (gt_baseline * 255).astype('uint8')
    contours= getContours(gt_baseline)
    debug_im = cv2.cvtColor(gt_baseline, cv2.COLOR_GRAY2BGR)
    slopes = []
    for cnt in contours:
        epsilon = 10 #0.01 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        points = sort_contours_hor(approx)
        prev_p = points[0]
        N = len(points) - 1
        for i, p in enumerate(points):
            cv2.line(debug_im, prev_p, p, (0, 255, 0), 2)
            if p[0] - prev_p[0] > 1:
                slopes.append(1.0 * abs(p[1] - prev_p[1]) / (p[0] - prev_p[0]))
            cv2.circle(debug_im, prev_p, 3, (0, 0, 255), -1)
            cv2.circle(debug_im, p, 4, (0,0,255), -1)
            if i not in [0, N]:
                cv2.circle(gt_baseline, p, 4, 0, -1)
            prev_p = p
    avg_slope = np.average(slopes)
    print('avg slope', avg_slope)
    force_straight_box = (avg_slope < 0.02)
    if force_straight_box: print('Enforcing straight lines')

    # cv2.imwrite('lol.png', debug_im)

    gt_baseline = gt_baseline > 0
    labels, objects = connected_components(gt_baseline)
    LOOKUP_RANGE_UP = 80

    line_boxes = []
    min_line_thres = 10
    line_height_thres = 4 #6

    for i, o in enumerate(objects):
        x1, x2 = o[1].start, o[1].stop
        y1, y2 = o[0].start, o[0].stop
        roi = (slice(safe_bound(y1 - LOOKUP_RANGE_UP), y2 + 1), slice(x1, x2))
        region = gt_area[roi]
        baseline_mask = labels[roi] == i + 1
        baseline_mask_full = gt_baseline[roi]

        line_bounding, line_box, line_mask = estimate_box_from_area_mask(region, baseline_mask, baseline_mask_full, force_straight_box = force_straight_box)
        line_box = order_points(line_box)
        [tl, tr, br, bl] = line_box
        line_h = (euclidean(tl, bl) + euclidean(tr, br)) / 2
        line_w = (euclidean(tl, tr) + euclidean(bl, br)) / 2

        padding_bottom = max(int(line_h * 0.15), 2) if line_h > 1 and 1.0 * line_w  / line_h > 1.5 else 1

        for i in range(len(line_box)):
            line_box[i][0] += x1
            line_box[i][1] += safe_bound(y1 - LOOKUP_RANGE_UP)
        line_box[2][1] += padding_bottom
        line_box[3][1] += padding_bottom

        line_bounding = [line_bounding[0] + x1, line_bounding[1] + safe_bound(y1 - LOOKUP_RANGE_UP), line_bounding[2] + x1, line_bounding[3] + safe_bound(y1 - LOOKUP_RANGE_UP)]
        line_h = line_bounding[3] - line_bounding[1]
        if line_h > line_height_thres:
            line_boxes.append((line_bounding, line_box, line_mask))

    return line_boxes

def line_segmentation_area(img, gt_baseline, gt_sep, gt_area, safe_segmentation = False):

    labels, objects = connected_components(gt_baseline)
    LOOKUP_RANGE_UP = 80

    line_boxes = []
    min_line_thres = 30 if safe_segmentation else 10
    line_height_thres = 12 if safe_segmentation else 6

    for i, o in enumerate(objects):
        if height(o) < 40 and (not safe_segmentation or width(o) > 10):
            x1, x2 = o[1].start, o[1].stop
            y1, y2 = o[0].start, o[0].stop
            roi = (slice(safe_bound(y1 - LOOKUP_RANGE_UP), y2 + 1), slice(x1, x2))
            roi_y1 = (slice(safe_bound(y1 - LOOKUP_RANGE_UP), y1 - 1), slice(x1, x2))
            region = gt_area[roi]
            baseline_mask = labels[roi] == i + 1

            distance_to_nearest_baseline = find_lower_line_bound(gt_baseline[roi_y1], min_line_thres=min_line_thres)
            height_from_area, line_mask = estimate_height_from_area_mask(region, baseline_mask)

            if distance_to_nearest_baseline > 4:
                line_h = min(distance_to_nearest_baseline, height_from_area) + 1
            else:
                line_h = height_from_area

            line_h = int(line_h)
            # if line_h > 25:
            if not safe_segmentation:
                padding_bottom = max(int(line_h * 0.22), 2)
            else:
                padding_bottom = 2
            # else:
            #     padding_bottom = max(int(line_h * 0.12), 2)
            line_mask_h = line_mask.shape[0]
            mask = line_mask[safe_bound(line_mask_h - line_h - 1):, :]

            if line_h > line_height_thres:
                line_boxes.append(([x1, y2 - mask.shape[0] + 2, x2, y2 + padding_bottom], mask))


    return line_boxes


def visualize_line_seg_hires(img, gt_baseline, gt_area, gt_sep = None, line_seg = False):

    out_im = img.copy() #np.zeros(img.shape[:2], dtype='uint8') #
    h, w = img.shape[:2]
    print('res', w, h)
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gt_baseline = cv2.resize(gt_baseline, (w, h))
    gt_area = cv2.resize(gt_area, (w, h))

    gt_baseline = skelet(gt_baseline, expand=False, expand_horizontal=True)
    # gt_baseline = gt_baseline > 80
    gt_area = gt_area > 150 #skelet(gt_sep, thres=50, expand=False, horizontal=True)
    gt_area = gt_area & (1 - gt_baseline)

    # gt_baseline = gt_baseline & np.bitwise_not(gt_baseline & gt_sep)

    # out_im[gt_sep > 0] = [255, 0, 0]

    if line_seg:
        lines = line_segmentation(img, gt_baseline, gt_area, show_im = out_im, gt_upper = gt_sep)

        for l, m in lines:
            x1, y1, x2, y2 = l
            region = out_im[y1:y2, x1:x2]
            # region[m > 0] = [0, 0, 255]
            region[m > 0] = 255
            # cv2.rectangle(out_im, (x1,y1), (x2,y2), (0,0,255), 1)

    # out_im[gt_area > 0] = [0, 0, 255]
    out_im[gt_baseline > 0] = [0, 255, 0]

    if gt_sep is not None:
        gt_sep = cv2.resize(gt_sep, (w, h))
        # gt_upper = skelet(gt_upper, expand=False, expand_horizontal=True)
        gt_sep = gt_sep > 90
        out_im[gt_sep > 0] = [255, 0, 0]

    return out_im, gt_area


def threshold_and_upscale_all_channel(hires_outs, img_shape):
    if len(hires_outs) > 3:
        gt_baseline, gt_sep, gt_area, gt_border = hires_outs
    else:
        gt_baseline, gt_sep, gt_area = hires_outs

    gt_baseline = threshold_and_upscale_map(img_shape, gt_baseline, skeletonize=True, threshold=80, expand=True)
    gt_sep = threshold_and_upscale_map(img_shape, gt_sep, threshold=80)
    gt_area = threshold_and_upscale_map(img_shape, gt_area, threshold=150)

    if len(hires_outs) > 3:
        gt_border = threshold_and_upscale_map(img_shape, gt_border, skeletonize=True, threshold=80)
        return (gt_baseline, gt_sep, gt_area, gt_border)
    else:
        return (gt_baseline, gt_sep, gt_area)

def visualize_line_seg_hires_area(img, gt_baseline, gt_sep, gt_area, gt_border = None, safe_segmentation = False, new_box = False):

    out_im = img.copy()
    h, w = img.shape[:2]
    print('res', w, h)

    # gt_baseline = threshold_and_upscale_map(img.shape, gt_baseline, skeletonize=True, threshold=100)
    # gt_sep = threshold_and_upscale_map(img.shape, gt_sep, threshold=80)
    # gt_area = threshold_and_upscale_map(img.shape, gt_area, threshold=150)

    if gt_border is not None:
        gt_baseline = gt_baseline & (1 - (gt_border))
    gt_area = gt_area & (1 - (gt_baseline))
    blank = np.zeros((h, w), dtype='uint8')

    if not new_box:
        lines = line_segmentation_area(out_im, gt_baseline, gt_sep, gt_area, safe_segmentation = safe_segmentation)
    else:
        lines = line_segmentation_area_with_rectification(out_im, gt_baseline, gt_sep, gt_area)

    print("Found {} lines".format(len(lines)))

    for i, L in enumerate(lines):
        if new_box:
            l, box, m = L
            cv2.drawContours(blank, [box], 0, 255, 1)
        else:
            l, m = L
            x1, y1, x2, y2 = l
            cv2.rectangle(blank, (x1, y1), (x2, y2), 255, 1)

    if new_box:
        lines_with_box = [(l, b) for l, b, m in lines]
        lines = [l for l, _, m in lines]
    else:
        lines = [l for l, m in lines]

    blank[gt_baseline > 0] = 255
    out_im[gt_area > 0] = [0,0,255]

    if not new_box:
        return out_im, blank, lines
    else:
        return out_im, blank, lines, lines_with_box

def draw_rects(img, rects, color, thickness = 1):
    for r in rects:
        cv2.rectangle(img, (r[0], r[1]), (r[2], r[3]), color, thickness)

def visualize_border_hires(img, gt_border, gt_area, save_dir='', show_text=True):

    out_im = img.copy()
    h, w = img.shape[:2]

    # gt_border = threshold_and_upscale_map(img.shape, gt_border, skeletonize=True, threshold=80)
    # remove underline
    underline_mask = r_opening(gt_border & gt_area, (1, 8))

    gt_border = gt_border & (1 - underline_mask)

    out_im[gt_border > 0] = [0, 255, 0]

    blank = np.zeros((h,w), dtype='uint8')

    gt_border_small = (gt_border * 255).astype('uint8')
    # gt_border_small = cv2.resize(gt_border_small, (int(w / scale_ratio), int(h / scale_ratio))) > 0
    ratio = 1.0 * h / w
    if ratio > 2:
        gt_border_small = resize_to_prefered_width(gt_border_small, 600)
    elif ratio > 1:
        gt_border_small = resize_to_prefered_width(gt_border_small, 1000)
    # else:
    #     gt_border_small = resize_to_prefered_height(gt_border_small, 1000)
    gt_border_small = gt_border_small > 0

    scale_ratio = 1.0 * w / gt_border_small.shape[1]

    cell_boxes = calculate_cell_cut(gt_border_small, out_im, scale=10, save_dir=save_dir)
    cell_contours = get_rects_by_contour((gt_border_small * 255).astype('uint8'))

    cell_boxes += cell_contours

    cells = []

    for c in cell_boxes:
        cell_pos = [c[0][1], c[0][0], c[1][1], c[1][0]]
        for i in range(len(cell_pos)):
            cell_pos[i] = int(cell_pos[i] * scale_ratio)
        cells.append(cell_pos)

    cells = filter_overlap_boxes_bigger(cells)
    # cells = list(reversed(cells))

    print("Found {} cells".format(len(cells)))

    for i, c in enumerate(cells):
        x1, y1, x2, y2 = c
        # cv2.rectangle(out_im, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.rectangle(blank, (x1, y1), (x2, y2), 255, 2)
        if show_text:
            cv2.putText(blank, str(i + 1), (x1, y2), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                        0.7, 255, 1, 2)

    # blank[gt_border > 0] = 255

    return out_im, blank, cells


def draw_mask(img, mask, color, opacity = 0.5):
    img_origin = img.copy()
    img[mask > 0] = color

    return cv2.addWeighted(img_origin, 1 - opacity, img, opacity, 0)


def check_invert(image):
    _, th3 = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    im_mean = np.mean(th3) / 255.0

    if im_mean < 0.45:
        image = np.bitwise_not(image)
    return image

def order_points(pts):
    pts = np.array(pts)
    # sort the points based on their x-coordinates
    xSorted = pts[np.argsort(pts[:, 0]), :]

    # grab the left-most and right-most points from the sorted
    # x-roodinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]

    # now, sort the left-most coordinates according to their
    # y-coordinates so we can grab the top-left and bottom-left
    # points, respectively
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost

    # now that we have the top-left coordinate, use it as an
    # anchor to calculate the Euclidean distance between the
    # top-left and right-most points; by the Pythagorean
    # theorem, the point with the largest distance will be
    # our bottom-right point
    # D = cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    # (br, tr) = rightMost[np.argsort(D)[::-1], :]
    (tr, br) = rightMost[np.argsort(rightMost[:, 1]), :]

    # return the coordinates in top-left, top-right,
    # bottom-right, and bottom-left order
    return np.array([tl, tr, br, bl], dtype="int64")

def pad_box(box, pad):
    pad_h = 1
    [tl, tr, br, bl] = box #order_points(box)
    tl[0] -= pad; tl[1] -= pad_h
    tr[0] += pad; tr[1] -= pad_h
    br[0] += pad; br[1] += pad
    bl[0] -= pad; bl[1] += pad

    return np.array([tl, tr, br, bl], dtype="float32")

def export_lines_to_disk(img, lines, cells, line_cell_id, dir_name, scale_factor, addition_json = {}, new_box = False):
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)

    dict_location = {}

    for i, c in enumerate(cells):
        c = scale_rect(c, scale_factor)
        x1, y1, x2, y2 = c
        dict_location["table1_cell%02d" % (i + 1)] = {'location': [y1, x1, y2, x2]}

    pad = 2
    for i, l in enumerate(lines):
        if new_box:
            l, box = l
        l = scale_rect(l, scale_factor)
        x1, y1, x2, y2 = l
        if not new_box:
            x2 += pad
            # y2 += pad
            x1 = safe_bound(x1 - pad)
            y1 = safe_bound(y1 - pad)

            line_im = img[y1:y2, x1:x2]
        else:
            box = scale_pts(box, scale_factor)
            box = pad_box(box, pad=pad)
            line_im = extract_line_warp(img, box)

        if len(line_im.shape) > 2:
            line_im = cv2.cvtColor(line_im, cv2.COLOR_BGR2GRAY)
        line_im = check_invert(line_im)
        cell_id = line_cell_id[i]
        im_name = "table1_cell%02d_line%02d" % (cell_id + 1, i + 1) if cell_id >= 0 else "text_line%02d" % (i + 1)
        # print(im_name)

        dict_location[im_name] = {'location': [y1, x1, y2, x2]}

        cv2.imwrite("{}/{}.png".format(dir_name, im_name), line_im)

    dict_location = {**dict_location, **addition_json}

    with open('%s/data.json' % dir_name, 'w') as fp:
        json.dump(dict_location, fp)

def extract_line_warp(im, box):
    # box = order_points(box)
    def l2(p1, p2):
        return np.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)

    w = int(max(l2(box[0], box[1]), l2(box[2],box[3])))+1
    h = int(max(l2(box[3], box[0]), l2(box[1],box[2])))+1

    poly = np.array([(0,0), (w-1,0), (w-1, h-1), (0, h-1)]).astype('float32')
    matrix = cv2.getPerspectiveTransform(box.astype('float32'), poly)
    return cv2.warpPerspective(im, matrix, (w, h))

def export_lines(img, lines, cells, line_cell_id, scale_factor, new_box=False):
    cells_with_meta = []

    for i, c in enumerate(cells):
        # c = scale_rect(c, scale_factor)
        x1, y1, x2, y2 = c
        cells_with_meta.append({'pos':[x1,y1,x2,y2], 'lines':[]})

    pad = 2
    for i, l in enumerate(lines):
        if new_box:
            l, box = l
        l = scale_rect(l, scale_factor)
        x1, y1, x2, y2 = l
        if not new_box:
            x2 += pad
            # y2 += pad
            x1 = safe_bound(x1 - pad)
            y1 = safe_bound(y1 - pad)

            line_im = img[y1:y2, x1:x2]
        else:
            box = scale_pts(box, scale_factor)
            box = pad_box(box, pad=pad)
            line_im = extract_line_warp(img, box)

        if len(line_im.shape) > 2:
            line_im = cv2.cvtColor(line_im, cv2.COLOR_BGR2GRAY)
        line_im = check_invert(line_im)
        cell_id = line_cell_id[i]
        if cell_id >= 0:
            cells_with_meta[cell_id]['lines'].append({'pos':[x1,y1,x2,y2], 'im':line_im})

    for i, cell in enumerate(cells_with_meta):
        x1, y1, x2, y2 = cell['pos']
        cells_with_meta[i]['lines'] = sort_cell_reading_order(cells_with_meta[i]['lines']) if (y2 - y1) / (x2 - x1) < 2.5 else []

    return cells_with_meta

def sort_cell_reading_order(cells_list):
   """ Sort cell list to create the right reading order using their locations

   :param cells_list: list of cells to sort
   :returns: a list of cell lists in the right reading order that contain no key or start with a key and contain no other key
   :rtype: list of lists of cells

   """
   sorted_list = []
   if len(cells_list) == 0:
       return cells_list

   while len(cells_list) > 1:
       topleft_cell = cells_list[0]
       for cell in cells_list[1:]:

           topleft_cell_pos = topleft_cell['pos']
           topleft_cell_center_x = (
                                       topleft_cell_pos[0] + topleft_cell_pos[2]) / 2
           topleft_cell_center_y = (
                                       topleft_cell_pos[1] + topleft_cell_pos[3]) / 2

           cell_pos = cell['pos']
           cell_center_x = (cell_pos[0] + cell_pos[2]) / 2
           cell_center_y = (cell_pos[1] + cell_pos[3]) / 2
           cell_h = cell_pos[3] - cell_pos[1]
           if cell_center_y <= topleft_cell_center_y - cell_h / 2:
               topleft_cell = cell
               continue
           if cell_center_x < topleft_cell_pos[1] and cell_center_y < topleft_cell_pos[2]:
               topleft_cell = cell
               continue

       sorted_list.append(topleft_cell)
       cells_list.remove(topleft_cell)
   sorted_list.append(cells_list[0])

   return sorted_list






