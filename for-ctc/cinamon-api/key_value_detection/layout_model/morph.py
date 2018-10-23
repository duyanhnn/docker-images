import os

from scipy.ndimage.filters import maximum_filter, minimum_filter
from scipy.ndimage.measurements import label as cc_label
from scipy.ndimage.measurements import find_objects
import cv2
import numpy as np
from math import ceil, floor
from skimage.filters import threshold_sauvola
from .cell_cut import cell_cut

def connected_components(image, thres = 0):

    if thres > 0:
        binary = image > thres
    else:
        binary = image

    labels, _ = cc_label(binary)
    objects = find_objects(labels)

    return labels, objects

def width(s):
    return s[1].stop-s[1].start

def height(s):
    return s[0].stop-s[0].start

def area(s):
    return (s[1].stop-s[1].start) * (s[0].stop-s[0].start)

def min_dim(b):
    if (width(b) > height(b)):
        return height(b)
    else:
        return width(b)

def max_dim(b):
    if (width(b) > height(b)):
        return width(b)
    else:
        return height(b)


def xcenter(s):
    return np.mean([s[1].stop, s[1].start])

def ycenter(s):
    return np.mean([s[0].stop, s[0].start])

def aspect_normalized(s):
    asp = height(s) * 1.0 / width(s)
    if asp < 1: asp = 1.0 / asp
    return asp

def r_dilation(image,size,origin=0):
    """Dilation with rectangular structuring element using maximum_filter"""
    return maximum_filter(image,size,origin=origin, mode='constant')

def r_erosion(image,size,origin=0):
    """Erosion with rectangular structuring element using maximum_filter"""
    return minimum_filter(image,size,origin=origin, mode='constant')

def r_opening(image,size,origin=0):
    """Opening with rectangular structuring element using maximum/minimum filter"""
    image = r_erosion(image,size,origin=origin)
    return r_dilation(image,size,origin=origin)

def r_closing(image,size,origin=0):
    """Closing with rectangular structuring element using maximum/minimum filter"""
    image = r_dilation(image,size,origin=0)
    return r_erosion(image,size,origin=0)


def compute_separators_morph_vertical(binary, scale, widen=True):
    """Finds vertical black lines corresponding to column separators."""
    span = 5  # min(5,  int(scale * 0.2))

    d0 = span
    if widen:
        d1 = span + 1
    else:
        d1 = span
    thick = r_dilation(binary, (d0, d1))
    vert = r_opening(thick, (int(2 * scale), 1))
    vert = r_erosion(vert, (d0 // 2, span))

    return vert


def compute_separators_morph_horizontal(binary, scale, widen=True):
    """Finds vertical black lines corresponding to column separators."""
    span = 5
    d0 = span  # int(max(5, scale / 5))
    if widen:
        d1 = span + 1
    else:
        d1 = span
    thick = r_dilation(binary, (d1, d0))
    hor = r_opening(thick, (1, int(2 * scale)))
    hor = r_erosion(hor, (span, d0 // 2))

    return hor


def compute_combine_seps(binary, scale):
    hor_seps = compute_separators_morph_horizontal(binary, scale)
    ver_seps = compute_separators_morph_vertical(binary, scale)
    combine_seps = hor_seps | ver_seps

    return combine_seps

## redraw combine sep using straight lines
def redraw_combine_sep(vertical_seps, horizontal_seps, scale, im_shape):

    h, w = im_shape

    # detect and re-draw vertical seps
    labels, _ = cc_label(horizontal_seps)
    objects = find_objects(labels)

    thickness_threshold = 20
    closeness = 0.4
    border_space = 8
    too_thick_thres = 6 #1.2
    skew_thres = 8

    hor_seps = np.zeros(horizontal_seps.shape, dtype='uint8')
    hor_sep_lines = []
    hor_sep_boxes = []

    for i, b in enumerate(objects):
        if height(b) < scale * thickness_threshold:
            y_center = int(ycenter(b))
            # skip line at page border
            # if width(b) > 0.4 * w and (y_center < border_space or y_center > h - border_space): #and (b[1].start == 0 or b[1].stop == w - 1):
            #     continue
            projection = np.sum(horizontal_seps[b], axis=0).astype(int)
            thickness = np.median(projection) - 1
            thickness_diff = np.amax(projection) - thickness

            # skip line if too thick
            if thickness > scale * too_thick_thres:
                continue
            # use bottom base line if too thick
            if thickness_diff > scale * 0.25:
                y_center = int(b[0].stop - thickness // 2)
            hor_sep_lines.append(((b[1].start + thickness, y_center), (b[1].stop - thickness, y_center), thickness))
            hor_sep_boxes.append(((b[1].start, y_center - int(ceil(thickness / 2.0))), (b[1].stop, y_center + int(floor(thickness / 2.0)))))

    # sort by y level
    hor_sep_lines = sorted(hor_sep_lines, key=lambda l : l[0][1])
    prev_y_pos = -1

    # merge and draw hor_sep
    for k, line in enumerate(hor_sep_lines):
        start, end, thickness = line

        if prev_y_pos > 0 and abs(start[1] - prev_y_pos) < closeness * scale:
            # merge with last y_pos
            start = (start[0], prev_y_pos)
            end = (end[0], prev_y_pos)
        else:
            prev_y_pos = start[1]

        #cv2.line(hor_seps, start, end, color=(255, 255, 255), thickness=thickness)
        start, end = hor_sep_boxes[k]
        hor_seps[start[1]:end[1] + 1, start[0]:end[0] + 1].fill(255)


    # detect and re-draw horizontal seps
    labels, _ = cc_label(vertical_seps)
    objects = find_objects(labels)

    ver_seps = np.zeros(vertical_seps.shape, dtype='uint8')
    ver_sep_lines = []
    ver_sep_boxes = []

    for i, b in enumerate(objects):
        if width(b) < scale * thickness_threshold:
            # skip line at page border
            # if height(b) > 0.4 * h and (x_center < border_space or x_center > w - border_space): # and (b[0].start == 0 or b[0].stop == h - 1):
            #     continue
            projection = np.sum(vertical_seps[b], axis=1).astype(int)
            thickness = np.median(projection) - 1
            thickness_diff = np.amax(projection) - thickness

            # skip line if too thick
            if thickness > scale * too_thick_thres:
                continue

            # if line is too skewed, break down to multiple parts:
            if thickness == 0 or 1.0 * width(b) / thickness > skew_thres * 5:
                continue

            child_seps = []
            if 1.0 * width(b) / thickness > skew_thres and width(b) > scale * 2 and height(b) > scale * 25:
                divide_step_count = int(ceil(1.0 * height(b) / scale / 10))
                x1, y1, x2, y2 = b[1].start, b[0].start, b[1].stop, b[0].stop
                if np.sum(vertical_seps[y1 : y1 + (y2-y1) // 4, x1 : x1 + (x2-x1) // 4]) > np.sum(vertical_seps[y1 : y1 + (y2-y1) // 4, x2 - (x2-x1) // 4 : x2]):
                    # start from top left
                    start_x, start_y = b[1].start, b[0].start
                    step_x, step_y = width(b) // divide_step_count, height(b) // divide_step_count
                else:
                    # start from top right
                    step_x, step_y = - (width(b) // divide_step_count), height(b) // divide_step_count
                    start_x, start_y = b[1].stop - step_x, b[0].start

                for i in range(divide_step_count):
                    child_seps.append((slice(start_y, start_y + step_y), slice(start_x, start_x + abs(step_x))))
                    start_y += step_y
                    start_x += step_x
            else:
                child_seps = [b]

            for c in child_seps:
                x_center = int(xcenter(c))
                # use left base line if too thick
                if thickness_diff > scale * 0.25:
                    x_center = int(c[1].start + thickness // 2)

                ver_sep_lines.append(((x_center, c[0].start + thickness), (x_center, c[0].stop - thickness), thickness))
                ver_sep_boxes.append(((x_center - int(ceil(thickness / 2.0)), c[0].start), (x_center + int(floor(thickness / 2.0)), c[0].stop)))

    # sort by y level
    ver_sep_lines = sorted(ver_sep_lines, key=lambda l: l[0][0])
    prev_x_pos = -1

    # merge and draw ver_sep
    for k, line in enumerate(ver_sep_lines):
        start, end, thickness = line

        # if prev_x_pos > 0 and abs(start[0] - prev_x_pos) < closeness * scale:
        #     # merge with last y_pos
        #     start = (prev_x_pos, start[1])
        #     end = (prev_x_pos, end[1])
        # else:
        #     prev_x_pos = start[0]

        start, end = ver_sep_boxes[k]
        ver_seps[start[1]:end[1] + 1, start[0]:end[0] + 1].fill(255)

        #cv2.line(ver_seps, start, end, color=(255, 255, 255),
                 #thickness=thickness)
        #cv2.rectangle(ver_seps, )

    combine_seps = np.bitwise_or(ver_seps, hor_seps)

    return combine_seps, hor_sep_boxes, ver_sep_boxes



# find table from edge image using connected components
def find_table_from_combine_sep(combine_sep, scale, maxsize):

    # using closing morphology to connect disconnected edges
    close_thes = int(scale * 0.4)
    closed_sep = r_dilation(combine_sep, (close_thes, close_thes))

    labels, _ = cc_label(closed_sep)
    objects = find_objects(labels)

    # result table list
    boxes = []

    for i, b in enumerate(objects):
         if width(b) > maxsize * scale or area(b) > scale * scale * 10 or (aspect_normalized(b) > 6 and max_dim(b) > scale * 1.5):

            density = np.sum(combine_sep[b])
            density = density / area(b)

            if (area(b) > scale * scale * 10 and min_dim(b) > scale * 1.0 and max_dim(b) > scale * 8 and density < 0.4):
                # calculate projection to determine table border
                w = width(b)
                h = height(b)

                region = (labels[b] == i + 1).astype('uint8')

                border_pad = max(w, h)
                border_thres = scale * 2

                proj_x = np.sum(region, axis=0)
                proj_y = np.sum(region, axis=1)

                proj_x[3:] += proj_x[:-3]
                proj_y[3:] += proj_y[:-3]

                sep_x = np.sort([j[0] for j in np.argwhere(proj_x > 0.75 * h)])
                sep_y = np.sort([j[0] for j in np.argwhere(proj_y > 0.75 * w)])

                # skip if sep count < 2
                if len(sep_x) < 1 or len(sep_y) < 1: continue

                border_left, border_right, border_top, border_bottom = None, None, None, None
                has_border_left, has_border_right, has_border_top, has_border_bottom = False, False, False, False

                if sep_x[0] < border_pad:
                    border_left = sep_x[0]
                    if sep_x[0] < border_thres:
                        has_border_left = True
                if sep_x[-1] > w - border_pad:
                    border_right = sep_x[-1]
                    if sep_x[-1] > w - border_thres:
                        has_border_right = True
                if sep_y[0] < border_pad:
                    border_top = sep_y[0]
                    if sep_y[0] < border_thres:
                        has_border_top = True
                if sep_y[-1] > h - border_pad:
                    border_bottom = sep_y[-1]
                    if sep_y[-1] > h - border_thres:
                        has_border_bottom = True

                #print_info(border_top, border_bottom, border_left, border_right)

                if all([j is not None for j in [border_top, border_bottom, border_left, border_right]]):
                    #boxes.append([b[1].start + border_left, b[0].start + border_top, b[1].start + border_right, b[0].start + border_bottom])
                    add_border = not (has_border_bottom or has_border_top or has_border_left or has_border_right) or width(b) > combine_sep.shape[1] * 0.8
                    boxes.append(([b[1].start, b[0].start, b[1].stop, b[0].stop], add_border))

    return boxes


def detect_small_table(region, scale):
    binary = region > 0
    h, w = region.shape
    #DSAVE('small_table', binary)
    ver_seps = compute_separators_morph_vertical(binary, scale = scale, widen = False)
    hor_seps = compute_separators_morph_horizontal(binary, scale = scale, widen=False)

    labels, _ = cc_label(hor_seps)
    objects = find_objects(labels)

    start_y, end_y = h, 0
    start_x, end_x = w, 0

    for i, o in enumerate(objects):
        if width(o) > w * 0.6:
            if start_y > ycenter(o):
                start_y = ycenter(o)
            if end_y < ycenter(o):
                end_y = ycenter(o)
            if o[1].stop - 4 > end_x:
                end_x = o[1].stop - 4

    if end_y - start_y < 90:
        end_y = h

    labels, _ = cc_label(ver_seps)
    objects = find_objects(labels)

    for i, o in enumerate(objects):
        if height(o) > h * 0.4:
            if start_x > xcenter(o):
                start_x = xcenter(o)
            if end_x < xcenter(o):
                end_x = xcenter(o)

    combine_seps = hor_seps | ver_seps
    # cv2.imwrite('small_table_seps.png', combine_seps * 255)

    y1, y2 = int(start_y - 1), int(end_y + 2)
    x1, x2 = int(start_x - 1), int(end_x + 2)

    result = binary[y1:y2, x1:x2]
    #DSAVE('out_table', result)

    cell_lists = []

    if min(y2 - y1, x2 - x1) < 8:
        return []

    step_y = (y2 - y1) / 2
    step_x = (x2 - x1) / 8

    for i in range(2):
        for j in range(8):
            y = i * step_y
            x = j * step_x
            cell_lists.append([int(x1 + x), int(y1 + y), int(x1 + x + step_x), int(y1 + y + step_y)])

    return cell_lists #[(x1, y1, x2, y2)]

def read_check_mark(cell_im):
    h, w = cell_im.shape[:2]
    cell_im = binarize_sauvola(cell_im, threshold=0.35)
    cell_im_bin = cell_im.copy()
    pad_x = 6
    pad_y = 3
    cell_im = cell_im[pad_y:-pad_y,pad_x:-pad_x]
    # cv2.imshow('lol', cell_im)
    # cv2.waitKey(500)
    binary = (cell_im == 0)
    thres = 4

    return np.sum(binary[:int(h * 0.27),:]) > thres and np.sum(binary[int(h * 0.73):,:]) > thres, cell_im_bin

def binarize_sauvola(image, threshold = 0.15):
    if len(image.shape) > 2 and image.shape[2] >= 3:
        convert_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        convert_image = image

    window_size = 25
    threshold = threshold_sauvola(convert_image, window_size=window_size, k=threshold)
    th3 = ((convert_image > threshold) * 255).astype('uint8')

    return th3

def filter_overlap_boxes(boxes, with_meta = False):
    if (len(boxes) < 2):
        return boxes

    is_overlap = [False] * len(boxes)

    for i in range(len(boxes)):
        for j in range(len(boxes)):
            if (i == j): continue
            if with_meta:
                x1, y1, x2, y2 = boxes[i][0]
                x3, y3, x4, y4 = boxes[j][0]
            else:
                x1, y1, x2, y2 = boxes[i]
                x3, y3, x4, y4 = boxes[j]
            if (is_overlap[j] == False and abs(x1-x2) <= abs(x3-x4)) and (x1 >= x3 and x2 <= x4 and y1 >= y3 and y2 <= y4):
                is_overlap[i] = True
                break

    return [boxes[i] for i in range(len(boxes)) if not is_overlap[i]]

def filter_overlap_boxes_bigger(boxes, with_meta = False):
    if (len(boxes) < 2):
        return boxes

    is_overlap = [False] * len(boxes)

    for i in range(len(boxes)):
        for j in range(len(boxes)):
            if (i == j): continue
            if with_meta:
                x1, y1, x2, y2 = boxes[i][0]
                x3, y3, x4, y4 = boxes[j][0]
            else:
                x1, y1, x2, y2 = boxes[i]
                x3, y3, x4, y4 = boxes[j]
            intersect_a = intersect_area(boxes[i], boxes[j])
            area_i = rect_area(boxes[i])
            area_j = rect_area(boxes[j])
            if (is_overlap[i] == False and ((area_i <= area_j) and intersect_a > 0.9 * min(area_j, area_i) and min(area_i,area_j) > 200) ):
                is_overlap[j] = True
                break

    return [boxes[i] for i in range(len(boxes)) if not is_overlap[i]]

def rect_area(rect):
    x1, y1, x2, y2 = rect
    return (x2 - x1) * (y2 - y1)

def intersect_area(box_a, box_b, min_thresh = 2):
    x1, y1, x2, y2 = box_a
    x3, y3, x4, y4 = box_b

    left, right = max(x1, x3), min(x2, x4)
    top, bottom = max(y1, y3), min(y2, y4)

    #print(box_a, box_b, left, right, top, bottom)

    if left < right - min_thresh  and top < bottom - min_thresh :
        #print(1.0 * (right - left) * (bottom - top))
        return 1.0 * (right - left) * (bottom - top)

    return 0.0

def check_intersect_boxes(boxes, scale):
    is_box_overlap = [False] * len(boxes)
    for i in range(len(boxes)):
        for j in range(len(boxes)):
            if (i == j): continue
            if intersect_area(boxes[i], boxes[j]) > scale * 15:
                is_box_overlap[i] = True
                break

    return is_box_overlap


#### check if big rectangle box contain smaller one
def is_overlap(big_box, small_box, pad=2):
    x1, y1, x2, y2 = small_box
    x3, y3, x4, y4 = big_box
    x3 -= pad
    y3 -= pad
    x4 += pad
    y4 += pad
    return (x1 >= x3 and x2 <= x4 and y1 >= y3 and y2 <= y4)


def scale_rect(rect, scale_factor):
    return [int(i * scale_factor) for i in rect]

def scale_pts(pts, scale_factor):
    return [[int(i * scale_factor) for i in pt] for pt in pts]

# get inner contours with rectangular shape to estimate cell
def get_4p_contours(image, threshold=-1):
    blobs = []
    topLevelContours = []

    _, contours, hierarchy = cv2.findContours(image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return []

    for i in range(len(hierarchy[0])):

        if len(contours[
                   i]) > 2:  # 1- and 2-point contours have a divide-by-zero error in calculating the center of mass.

            # bind each contour with its corresponding hierarchy context description.
            obj = {'contour': contours[i], 'context': hierarchy[0][i]}
            blobs.append(obj)

    for blob in blobs:
        child = blob['context'][2]
        parent = blob['context'][3]
        if child <= threshold and parent != -1:  # no parent, therefore a root
            c = blob['contour']
            # approximate the contour
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)

            # if our approximated contour has four points
            if len(approx) == 4:
                topLevelContours.append(c)

    return topLevelContours

#### filter horizontal and vertical separators outside of table regions
def filter_seps_by_table_regions(hor_seps, ver_seps, table_regions, is_table_box_overlap, redrawed_combine_seps, scale, add_table_border = True):
    filtered_hor_seps = []
    filtered_ver_seps = []

    for table, _ in table_regions:
        for sep in hor_seps:
            if is_overlap(table, [sep[0][0], sep[0][1], sep[1][0], sep[1][1]]):
                filtered_hor_seps.append(sep)

        for sep in ver_seps:
            if is_overlap(table, [sep[0][0], sep[0][1], sep[1][0], sep[1][1]]):
                filtered_ver_seps.append(sep)

    if add_table_border:
        # thickness for additional table border
        thickness = 3
        for i, table in enumerate(table_regions):
            x1, y1, x2, y2 = table[0]
            add_border = table[1]
            if is_table_box_overlap[i] or min(x2-x1, y2-y1) < scale * 3: continue
            x1, y1, x2, y2 = x1 + 2, y1 + 2, x2 - 2, y2 - 2

            if add_border:

                # additional vertical separators
                filtered_ver_seps.append(((x1, y1), (x1 + thickness, y2)))
                filtered_ver_seps.append(((x2 - thickness, y1), (x2, y2)))

                redrawed_combine_seps[y1:y2, x1 : x1 + thickness].fill(255)
                redrawed_combine_seps[y1:y2, x2 - thickness : x2].fill(255)

                # additional horizontal separators
                filtered_hor_seps.append(((x1, y1), (x2, y1 + thickness)))
                filtered_hor_seps.append(((x1, y2 - thickness), (x2, y2)))

                redrawed_combine_seps[y1: y1 + thickness , x1:x2].fill(255)
                redrawed_combine_seps[y2 - thickness: y2 , x1:x2].fill(255)

    return filtered_hor_seps, filtered_ver_seps

def get_rects_by_contour(image):
    contours = get_4p_contours(image)
    boxes = []
    for contour in contours:
        box = cv2.boundingRect(contour)
        # boxes.append([box[1], box[0], box[1] + box[3], box[0] + box[2]])
        boxes.append([(box[1], box[0]), (box[1] + box[3], box[0] + box[2])])
    return boxes

def getHoughLinesP(im, min_length = 100, max_line_gap = 10):
    debug_im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    minLineLength = min_length
    maxLineGap = max_line_gap
    lines = cv2.HoughLinesP(im,1,np.pi/180,100,minLineLength,maxLineGap)
    for l in lines:
        x1, y1, x2, y2 = l[0]
        cv2.line(debug_im,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.circle(debug_im, (x1, y1), 3, (0,0,255), -1)
        cv2.circle(debug_im, (x2, y2), 3, (0, 0, 255), -1)
    return [l[0] for l in lines], debug_im

def getContours(sourceImage, threshold=-1):

    image = sourceImage.copy()
    blobs = []
    topLevelContours = []

    try:

        _, contours, hierarchy = cv2.findContours(
            image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        for i in range(len(hierarchy[0])):

            if len(contours[i]) > 0:
                # 1- and 2-point contours have a divide-by-zero error
                # in calculating the center of mass.

                # bind each contour with its corresponding hierarchy
                # context description.
                obj = {'contour': contours[i], 'context': hierarchy[0][i]}
                blobs.append(obj)

        for blob in blobs:
            parent = blob['context'][3]
            if parent <= threshold: # no parent, therefore a root
                topLevelContours.append(blob['contour'])

    except TypeError as e:
        pass

    return topLevelContours

from scipy.spatial.distance import euclidean
def sort_contours_hor(cnt):
    points = np.array([p[0] for p in cnt])
    sorted_points = points[np.argsort([p[0] for p in points])]
    filtered_points = [tuple(sorted_points[0])]
    prev_p = sorted_points[0]
    dist_thres = 20
    for p in sorted_points[1:]:
        if euclidean(p, prev_p) > dist_thres:
            filtered_points.append(tuple(p))
            prev_p = p
    return filtered_points

def calculate_cell_cut(border_mask, origin_im, scale, save_dir=''):
    hor_sep = compute_separators_morph_horizontal(border_mask, scale)
    ver_sep = compute_separators_morph_vertical(border_mask, scale)

    # redraw combine sep with straight line
    redrawed_combine_seps, hor_sep_boxes, ver_sep_boxes = redraw_combine_sep(ver_sep, hor_sep, scale, border_mask.shape)

    cv2.imwrite(os.path.join(save_dir, "_combine_sep_redraw.png"), redrawed_combine_seps)

    # detect table from combine seps
    table_boxes = find_table_from_combine_sep(redrawed_combine_seps > 0, scale, scale * 4)
    table_boxes = filter_overlap_boxes(table_boxes, with_meta=True)
    is_table_box_overlap = check_intersect_boxes([t[0] for t in table_boxes], scale)

    print("Found {} tables...".format(len(table_boxes)))

    hor_sep_boxes, ver_sep_boxes = filter_seps_by_table_regions(hor_sep_boxes, ver_sep_boxes, table_boxes,
                                                                is_table_box_overlap,
                                                                redrawed_combine_seps, scale,
                                                                add_table_border=True)
    table_boxes = [t for t, _ in table_boxes]

    list_boxes, debug_image = cell_cut(redrawed_combine_seps, origin_im, scale, table_boxes,
                                                  hor_sep_boxes, ver_sep_boxes, use_advance_cut=False, save_dir=save_dir)

    return list_boxes

def assign_line_to_cell(lines, cells, blank_shape, show_text=True):

    blank = np.zeros(blank_shape[:2], dtype='uint8')
    line_cell_id = [-1] * len(lines)
    line_cell_intersect_area = [0] * len(lines)

    for i, l in enumerate(lines):
        line_area = rect_area(l)
        for k, c in enumerate(cells):
            inter_area = intersect_area(l, c)
            if inter_area > line_area * 0.5 and inter_area > line_cell_intersect_area[i]:
                line_cell_id[i] = k
                line_cell_intersect_area[i] = inter_area

        if show_text:
            cv2.putText(blank, "{}_{}".format(str(line_cell_id[i] + 1), str(i + 1)), (l[2], l[3]), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                        0.7, 255, 1, 2)

    return line_cell_id, blank





