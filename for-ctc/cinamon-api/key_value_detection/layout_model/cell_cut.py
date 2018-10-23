import cv2
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import re
import operator

sum_point = [[]]
hlines = []
vlines = []
data = []
data2 = []
isCut = []

TX, TY = 1, 1  # diem nhieu gan line
DX, DY = 4, 4  # khoang cach giua 2 line
MIN_LINE_WIDTH = 6  # do rong toi thieu cua 1 line
MIN_CELL_X = 10  # do rong toi thieu cua 1 cell ( ngang )
MIN_CELL_Y = 15  # do rong toi thieu cua 1 cell ( doc )
MIN_CELL_XY = 20  # do rong toi thieu cua max 2 canh
MAX_LINE_NUM = 400  # so luong line cat ra
MAX_LINE_NUM_X = 81  # so luong line cat ra theo chieu ngang
MAX_LINE_NUM_Y = 62  # so luong line cat ra theo chieu doc
SCALE = 10
NOISE_THRESHOLD = 0.000001
LINE_HISTOGRAM_MIN = 0.5
LINE_HISTOGRAM_MAX = 1.2
PreK = 0

def clip_index(a, n, m):
    new_a = [a[0], a[1]]

    if new_a[0] >= n:
        new_a[0] = n-1
    if new_a[0] < 0:
        new_a[0] = 0
    if new_a[1] >= m:
        new_a[1] = m-1
    if new_a[1] < 0:
        new_a[1] = 0

    return new_a

def get_points(a, b):
    global data
    n, m = sum_point.shape
    a = clip_index(a, n, m)
    b = clip_index(b, n, m)

    #result = np.sum(data[a[0]:b[0] + 1, a[1]:b[1] + 1]) / 255.0
    result = sum_point[b[0]][b[1]] + sum_point[a[0] - 1][a[1] - 1] - sum_point[a[0] - 1][b[1]] - sum_point[b[0]][a[1] - 1]
    result = result / 255.0

    return result


def is_close(li, value, d):
    for l in li:
        if l[0] - d < value < l[1] + d:
            return True
    return False


def expand_same_den(pos, den):
    n, l, h = len(den), pos, pos
    while l > 1:
        diff = den[l - 1] / (1.0 * den[pos])
        if LINE_HISTOGRAM_MIN > diff or diff > LINE_HISTOGRAM_MAX:
            break
        l -= 1
    while h < n - 1:
        diff = den[h + 1] / (1.0 * den[pos])
        if LINE_HISTOGRAM_MIN > diff or diff > LINE_HISTOGRAM_MAX:
            break
        h += 1
    return l, h


def is_okay(box):
    global isCut
    global PreK
    global MIN_LINE_WIDTH
    hline_1 = hlines[box[0][0]][1]
    vline_1 = vlines[box[0][1]][1]
    hline_2 = hlines[box[1][0] + 1][0]
    vline_2 = vlines[box[1][1] + 1][0]

    is_cut = np.any(isCut[box[0][0] : box[1][0] + 1, box[0][1] : box[1][1]+ 1])

    # for x in range(box[0][0], box[1][0] + 1):
    #     for y in range(box[0][1], box[1][1]+ 1):
    #         if isCut[x][y] > 0:
    #             is_cut = True
    #             break
    #     if is_cut: break


    if hline_2 - hline_1 < MIN_CELL_X or vline_2 - vline_1 < MIN_CELL_Y:
        #k = 0
        return False
    else:
        k = get_points((hline_1 + DX, vline_1 + DY), (hline_2 - DX, vline_2 - DY))

    alpha = min(vline_2 - vline_1, hline_2 - hline_1) #max(min(vline_2 - vline_1, hline_2 - hline_1), SCALE)
    delta = 0 #k #- PreK
    #PreK = k
    return k >= 0 and (k < alpha * 0.5 * MIN_LINE_WIDTH) and (delta <  alpha * NOISE_THRESHOLD) and not is_cut

def is_ok_extend(box, direction):
    hline_1 = hlines[box[0][0]][1]
    vline_1 = vlines[box[0][1]][1]
    hline_2 = hlines[box[1][0] + 1][0]
    vline_2 = vlines[box[1][1] + 1][0]

    return not (direction[0] and hline_2 - hline_1 < 0) and not (direction[1] and vline_2 - vline_1 < 0)

def is_okay_border(box, border=[1, 1, 1, 1]):
    hline_1_1 = hlines[box[0][0]][0]
    vline_1_1 = vlines[box[0][1]][0]
    hline_1_2 = hlines[box[0][0]][1]
    vline_1_2 = vlines[box[0][1]][1]
    hline_2_1 = hlines[box[1][0] + 1][0]
    vline_2_1 = vlines[box[1][1] + 1][0]
    hline_2_2 = hlines[box[1][0] + 1][1]
    vline_2_2 = vlines[box[1][1] + 1][1]

    if hline_2_1 - hline_1_2 < MIN_CELL_X:
        return False
    if vline_2_1 - vline_1_2 < MIN_CELL_Y:
        return False
    if max(hline_2_1 - hline_1_2, vline_2_1 - vline_1_2) < MIN_CELL_XY:
        return False

    k = [0, 0, 0, 0]
    k[0] = 1.0 * get_points((hline_1_1, vline_1_2), (hline_1_2, vline_2_1)) / (vline_2_1 - vline_1_2 + 1) / (
        hline_1_2 - hline_1_1 + 1)
    k[2] = 1.0 * get_points((hline_2_1, vline_1_2), (hline_2_2, vline_2_1)) / (vline_2_1 - vline_1_2 + 1) / (
        hline_2_2 - hline_2_1 + 1)
    k[3] = 1.0 * get_points((hline_1_2, vline_1_1), (hline_2_1, vline_1_2)) / (hline_2_1 - hline_1_2 + 1) / (
        vline_1_2 - vline_1_1 + 1)
    k[1] = 1.0 * get_points((hline_1_2, vline_2_1), (hline_2_1, vline_2_2)) / (hline_2_1 - hline_1_2 + 1) / (
        vline_2_2 - vline_2_1 + 1)

    h = [vline_2_1 - vline_1_2 + 1, hline_2_1 - hline_1_2 + 1, vline_2_1 - vline_1_2 + 1, hline_2_1 - hline_1_2 + 1]
    l = [hline_1_2 - hline_1_1 + 1, hline_2_2 - hline_2_1 + 1, vline_1_2 - vline_1_1 + 1, vline_2_2 - vline_2_1 + 1]

    for i in range(4):
        if border[i] and (k[i] < 0.33 or (h[i] < MIN_CELL_XY and k[i] < 0.55) or (l[i] > 8 and k[i] < 0.5)):  # old param 0.33
            return False

    if max(k[0], k[2]) < 0.6:
        return False
    if max(k[1], k[3]) < 0.6:
        return False

    k = get_points((hline_1_1, vline_1_1), (hline_2_2, vline_2_2)) - get_points(
        (hline_1_2 + TX, vline_1_2 + TY), (hline_2_1 - TX, vline_2_1 - TY))

    if sum(border) == 4:
        return k > ((hline_2_2 - hline_1_1) * (border[1] + border[3]) + (vline_2_2 - vline_1_1) * (
            border[0] + border[2])) * 0.5
    else:
        return True


def get_point_box(box):
    hline_1 = hlines[box[0][0]][1]
    vline_1 = vlines[box[0][1]][1]
    hline_2 = hlines[box[1][0] + 1][0]
    vline_2 = vlines[box[1][1] + 1][0]
    return abs(get_points((hline_1 + DX, vline_1 + DY), (hline_2 - DX, vline_2 - DY)))


def get_box(box):
    a = (hlines[box[0][0]][1] + TX, vlines[box[0][1]][1] + TY)
    b = (hlines[box[1][0] + 1][0] - TX, vlines[box[1][1] + 1][0] - TY)
    return (a, b)


def cal_sum_matrix(data):
    n, m = data.shape

    sum_pt = np.cumsum(data, axis=0)
    sum_pt = np.cumsum(sum_pt, axis=1)

    # sum_pt = [[]]
    # sum_pt[0].append([])
    # sum_pt[0][0] = data[0][0]
    # for i in range(1, m):
    #     sum_pt[0].append([])
    #     sum_pt[0][i] = sum_pt[0][i - 1] + data[0][i]
    # for i in range(1, n):
    #     sum_pt.append([])
    #     sum_pt[i].append([])
    #     sum_pt[i][0] = sum_pt[i - 1][0] + data[i][0]
    #     for j in range(1, m):
    #         t = data[i][j]
    #         sum_pt[i].append(sum_pt[i][j - 1] + sum_pt[i - 1][j] + t - sum_pt[i - 1][j - 1])

    return sum_pt


def sort_and_remove_overlap(lines, max_line):
    ans = []
    lines.sort(key=lambda x: x[0])
    for id in range(len(lines) - 1):
        if lines[id][1] > lines[id + 1][0]:
            l = lines[id][1]
            h = lines[id + 1][0]
            lines[id] = (lines[id][0], l)
            lines[id + 1] = (h, lines[id + 1][1])
    for line in lines:
        if line[1] - line[0] < max_line:
            ans.append(line)
    return ans


def get_lines(l, d, den, max_len, size):
    ans = [expand_same_den(l[0][0], den)]
    for i in range(1, len(l)):
        if len(ans) > size or l[i][1] < 1:
            break
        if not is_close(ans, l[i][0], d):
            ans.append(expand_same_den(l[i][0], den))
    ans = sort_and_remove_overlap(ans, max_len)
    return ans


def get_hline_endpoints(line):
    start = 1
    n, m = data.shape
    ans = []
    while get_points((line[0], start), (line[1], m - 1)) > 0:
        low = start
        high = m - 1
        while low < high:
            mid = (low + high) / 2
            if get_points((line[0], start), (line[1], mid)) == 0:
                low = mid + 1
            else:
                high = mid
        s = low
        if s > m - 5:
            break
        for e in range(s, m - 3):
            if get_points((line[0], e + 1), (line[1], e + 3)) == 0:
                break
        if e - s > MAX_LINE_NUM_Y:
            ans.append((s, e))
        start = e + 1
        if start > m - 5:
            break
    return ans


def extend_cell(box, direction, boundary):
    global PreK
    i, j = box[0][0], box[0][1]
    u, v = box[1][0], box[1][1]
    PreK = get_point_box(box)
    while u < boundary[0] + (1 - direction[0]) and v < boundary[1] + (1 - direction[1]):
        new_box = ((i, j), (u + direction[0], v + direction[1]))
        if not is_okay(new_box) and is_ok_extend(new_box, direction):
            break
        u += direction[0]
        v += direction[1]

    if u >= boundary[0] + (1 - direction[0]):
        u = box[1][0]

    if v >= boundary[1] + (1 - direction[1]):
        v = box[1][1]

    return u, v

def first_vertical_extend(i, j, boundary):
    global isCut

    HEIGH_THRESH = 20

    if j >= boundary or isCut[i][j+1]:
        return j

    v = j
    while v < boundary + 1:
        hline_1 = hlines[i][1]
        vline_1 = vlines[v+1][0]
        hline_2 = hlines[i][1] + HEIGH_THRESH
        vline_2 = vlines[v+1][1]
        if get_points((hline_1, vline_1), (hline_2, vline_2)) > HEIGH_THRESH * (vline_2 - vline_1 + 1) * 0.6 and vlines[v+1][0] - vlines[j][1] > 20:
            #print(i,j,v,get_points((hline_1, vline_1), (hline_2, vline_2)), hline_2 - hline_1, vline_2 - vline_1)
            break
        v += 1

    if v > boundary: v = j

    return v

def first_horizontal_extend(i, j, boundary):
    global isCut

    if i >= boundary or isCut[i+1][j]:
        return i

    WIDTH_THRESH = 20

    u = i
    while u < boundary + 1:
        hline_1 = hlines[u + 1][0]
        vline_1 = vlines[j][1]
        hline_2 = hlines[u + 1][1]
        vline_2 = vlines[j][1] + WIDTH_THRESH
        if get_points((hline_1, vline_1), (hline_2, vline_2)) > WIDTH_THRESH * (hline_2 - hline_1 + 1) * 0.6 and hlines[u+1][0] - hlines[i][1] > 20:
            break
        u += 1

    if u > boundary: u = i

    return u

def get_full_border_cell(hsize, vsize):
    global isCut
    global PreK
    boxes = []
    for i in range(hsize - 1):
        for j in range(vsize - 1):
            if isCut[i][j] != 0:
                continue
            # if len(boxes) == 2:
            #     print
            # if i >= 12 and j >= 10:
            #     print

            u, v = i, j
            v = first_vertical_extend(i, j, vsize - 2)
            u = first_horizontal_extend(i, j, hsize - 2)

            # u, v = extend_cell(((i, j), (i, j)), (1, 0), (hsize - 2, vsize - 2))
            # u, v = extend_cell(((i, j), (u, v)), (0, 1), (hsize - 2, vsize - 2))
            ok = False

            while not (is_okay_border(((i, j), (u, v)), border=[1, 0, 0, 0]) and is_okay(((i, j), (u, v)))) and u > i:
                u -= 1
            while not (is_okay_border(((i, j), (u, v)), border=[0, 1, 0, 0]) and is_okay(((i, j), (u, v)))) and v > j:
                v -= 1

            if is_okay_border(((i, j), (u, v)), border=[1,1,1,1]) and is_okay(((i, j), (u, v))):
                ok = True

            # else:
            #     # search for different direction
            #     u, v = extend_cell(((i, j), (i, j)), (0, 1), (hsize - 2, vsize - 2))
            #     u, v = extend_cell(((i, j), (u, v)), (1, 0), (hsize - 2, vsize - 2))
            #
            #     while not is_okay_border(((i, j), (u, v))) and u > i:
            #         u -= 1
            #     while not is_okay_border(((i, j), (u, v))) and v > j:
            #         v -= 1
            #
            #     if is_okay_border(((i, j), (u, v))):
            #         ok = True

            if ok:
                #print("CELL ", len(boxes))
                boxes.append(((i, j), (u, v)))
                #print(((i, j), (u, v)))
                isCut[i:u+1, j:v+1].fill(len(boxes))
                # for x in range(i, u + 1):
                #     for y in range(j, v + 1):
                #         isCut[x][y] = len(boxes)

    return boxes


def get_cell_advance(hsize, vsize, boxes):
    global isCut
    global PreK

    new_boxes = []

    for i in range(hsize - 1):
        for j in range(vsize - 1):
            if isCut[i][j] != 0:
                continue
            u, v = extend_cell(((i, j), (i, j)), (1, 0), (hsize - 2, vsize - 2))
            if not is_okay(((i, j), (u, v))):
                continue
            PreK = get_point_box(((i, j), (u, v)))
            while v < vsize - 2:
                new_box = ((i, j), (u, v + 1))
                if not is_okay_border(((i, j), (u, v)), [1, 0, 1, 0]) or not is_okay(new_box):
                    break
                v += 1
            if is_okay_border(((i, j), (u, v)), [1, 0, 1, 0]):
                new_boxes.append(((i, j), (u, v)))
                for x in range(i, u + 1):
                    for y in range(j, v + 1):
                        isCut[x][y] = len(boxes) + len(new_boxes)
    return new_boxes


def is_overlap(big_box, small_box, pad=2):
    x1, y1, x2, y2 = small_box
    x3, y3, x4, y4 = big_box
    x3 -= pad
    y3 -= pad
    x4 += pad
    y4 += pad
    return (x1 >= x3 and x2 <= x4 and y1 >= y3 and y2 <= y4)


def get_list_line(boxes, position, n):
    lines = dict()
    # for box in boxes:
    #     u = int(box[0][position])
    #     if u in lines:
    #         lines[u] = lines[u] | 1
    #     else:
    #         lines[u] = 1
    #     v = int(box[1][position])
    #     #print(u, v)
    #     if v > n - 2:
    #         v = n - 2
    #     if v in lines:
    #         lines[v] = lines[v] | 2
    #     else:
    #         lines[v] = 2
    # sorted_lines = sorted(lines.iteritems(), key=operator.itemgetter(0))
    # #print(sorted_lines)
    # start = -1
    # ans_lines = []
    # for i in range(len(sorted_lines)):
    #     type = sorted_lines[i][1]
    #     if start == -1:
    #         if type == 1:
    #             start = sorted_lines[i][0]
    #         else:
    #             #print("warning cellcut ----")
    #             pass
    #         continue
    #     if type == 2:
    #         ans_lines.append((start, sorted_lines[i][0]))
    #         start = -1
    #     elif type == 1:
    #         ans_lines.append((start, sorted_lines[i][0] - 1))
    #         start = sorted_lines[i][0]
    #     elif type == 3:
    #         ans_lines.append((start, sorted_lines[i][0]))
    #         start = sorted_lines[i][0] + 1

    ans_lines = []
    for box in boxes:
        u = int(box[0][position])
        v = int(box[1][position])
        #print(u, v)
        if v > n - 2:
            v = n - 2
        ans_lines.append((u, v))

    ans_lines = sorted(ans_lines, key=operator.itemgetter(0))


    return ans_lines


def cell_cut(combine_seps, original, text_scale, table_boxes, hor_sep_boxes, ver_sep_boxes, use_advance_cut = True, save_dir=''):
    global sum_point
    global hlines
    global vlines
    global data2
    global PreK
    global data
    #global MAX_NOISE_POINT
    global SCALE
    global MIN_LINE_WIDTH
    global MAX_LINE_NUM
    global isCut

    ### debug code to dump input params to file
    # import pickle
    # pickle.dump((text_scale, table_boxes, hor_sep_boxes, ver_sep_boxes), open('debug.pkl', 'wb'))
    # return

    MIN_LINE_WIDTH = 4
    SCALE = int(text_scale)
    #MAX_NOISE_POINT = int(SCALE * 1.1)
    MAX_LINE_NUM = SCALE * 100

    MAX_LINE_X = SCALE / 1.2  # do day toi da cua 1 line ngang
    MAX_LINE_Y = SCALE / 0.6  # do day toi da cua 1 line doc

    data = np.asarray(combine_seps, dtype="int32")
    n, m = data.shape
    print('Calculating DP sum matrix')
    sum_point = np.array(cal_sum_matrix(data))

    hlines = get_list_line(hor_sep_boxes, 1, n)
    #print(hlines)
    hsize = len(hlines)

    print('N M', n, m)
    if hsize == 0:
        print('HSIZE = 0')
        return [], None

    # buf = []
    # density = [0]
    #
    # for i in range(1, m - 1):
    #     sumV = get_points((1, i), (hlines[0][0] - 1, i))
    #     for j in range(hsize - 1):
    #         if hlines[j][1] + 1 < hlines[j + 1][0]:
    #             sumV += get_points((hlines[j][1] + 1, i), (hlines[j + 1][0] - 1, i))
    #     sumV += get_points((hlines[hsize - 1][1], i), (n - 1, i))
    #     buf.append((i, sumV))
    #     density.append(sumV)
    #
    # buf = sorted(buf, key=lambda a: -a[1])
    # buf = buf[:MAX_LINE_NUM]

    vlines = get_list_line(ver_sep_boxes, 0, m)
    #vlines = get_lines(buf, DY, density, MAX_LINE_Y, MAX_LINE_NUM_Y)
    vsize = len(vlines)

    #### virtual lines
    # endpoints = dict()
    #
    # for hbox in hor_sep_boxes:
    #     u = int(hbox[0][0] / 2.0) * 2
    #     if u == 0:
    #         u = 1
    #     if u in endpoints:
    #         endpoints[u] += 1
    #     else:
    #         endpoints[u] = 1
    #     v = int(hbox[1][0] / 2.0) * 2
    #     if v > m - 3:
    #         v = m - 3
    #     if v in endpoints:
    #         endpoints[v] += 1
    #     else:
    #         endpoints[v] = 1
    # sort_points = sorted(endpoints.items(), key=operator.itemgetter(0))
    # # print(sort_points)
    # if vsize < 20:
    #     for point in sort_points:
    #         if point[1] > 1:
    #             inside = True
    #             for vline in vlines:
    #                 if vline[0] <= point[0] and point[0] <= vline[1]:
    #                     inside = False
    #             if inside:
    #                 vlines.append((point[0], point[0]))
    #             vlines = sort_and_remove_overlap(vlines, MAX_LINE_Y)
    # vsize = len(vlines)
    # print("Line num: ", hsize, vsize)

    #print(vlines)

    if hsize < 2 or vsize < 2:
        return [], None

    isCut = np.zeros((hsize - 1, vsize - 1), dtype=np.int)
    boxes = get_full_border_cell(hsize, vsize)
    MIN_LINE_WIDTH = 7

    if use_advance_cut:
        half_boxes = get_cell_advance(hsize, vsize, boxes)
    else:
        half_boxes = []

    if len(original.shape) == 2:
        # cv2.normalize(original, 0, 255, cv2.NORM_MINMAX)
        # original = (255 - original).astype('uint8')
        data2 = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
    else:
        data2 = np.array(original, copy=True)

    # check overlap cells
    # is_redundant = np.full((len(boxes)), False, dtype=bool)
    # for i, box1 in enumerate(boxes):
    #     box_pos = get_box(box1)
    #     cell_pos1 = [box_pos[0][1], box_pos[0][0], box_pos[1][1], box_pos[1][0]]
    #     area1 = (box_pos[1][1] - box_pos[0][1]) * (box_pos[1][0] - box_pos[0][0])
    #     for j, box2 in enumerate(boxes):
    #         box_pos = get_box(box2)
    #         cell_pos2 = [box_pos[0][1], box_pos[0][0], box_pos[1][1], box_pos[1][0]]
    #         area2 = (box_pos[1][1] - box_pos[0][1]) * (box_pos[1][0] - box_pos[0][0])
    #         if area2 < area1:
    #             continue
    #         if area1 == area2:
    #             if i < j:
    #                 if is_overlap(cell_pos2, cell_pos1, 2):
    #                     is_redundant[j] = True
    #             continue
    #         if is_overlap(cell_pos2, cell_pos1, text_scale):
    #             is_redundant[i] = True

    pos_boxes = []
    combined_boxes = boxes + half_boxes
    for i, box in enumerate(combined_boxes):
        #if not is_redundant[i]:
        box_pos = get_box(box)
        pos_boxes.append(box_pos)
        # for table in table_boxes:
        #     if is_overlap(table, cell_pos):
        #         pos_boxes.append(box_pos)
        #         break

    # new_pos_boxes = []
    # for i, box in enumerate(half_boxes):
    #     # if not is_redundant[i]:
    #     box_pos = get_box(box)
    #     new_pos_boxes.append(box_pos)
    #     # for table in table_boxes:
    #     #     if is_overlap(table, cell_pos):
    #     #         new_pos_boxes.append(box_pos)
    #     #         break

    # index = 1
    # for i, box in enumerate(pos_boxes):
    #     cv2.rectangle(data2, (box[0][1], box[0][0]),
    #                   (box[1][1], box[1][0]), (0, 255, 0), 2)
    #     debug_txt = ' ' + str(combined_boxes[i])
    #     cv2.putText(data2, str(index), (box[0][1], box[1][0]), cv2.FONT_HERSHEY_SIMPLEX,
    #                 1, (0, 0, 255), 2, 2)
    #     index += 1

    #### debug hlines and vlines

    # for box in table_boxes:
    #     cv2.rectangle(data2, (box[0], box[1]),
    #                   (box[2], box[3]), (0, 0, 255), 2)

    # for i in hlines:
    #     for j in range(1, m):
    #         data2[i[0]][j][2] = 255
    #         data2[i[1]][j][1] = 255
    # for i in vlines:
    #     for j in range(1, n):
    #         data2[j][i[0]][2] = 255
    #         data2[j][i[1]][1] = 255

    # import pickle
    # pickle.dump((combine_seps, pos_boxes, text_scale, table_boxes, hor_sep_boxes, ver_sep_boxes), open('template.pkl', 'wb'))


    cv2.imwrite(os.path.join(save_dir, '_debug_main2.png'), data2)

    return pos_boxes, data2


if __name__ == "__main__":
    filename = "/home/taprosoft/Downloads/test_segmented/flax_bprost/run/data_zip/Sony_Sonpo/Smartphone TEST 1/_combine_sep_redraw.png"
    image = np.array(Image.open(filename))
    import os
    path = os.path.dirname(filename)
    print('path', path)
    import pickle
    text_scale, table_boxes, hor_sep_boxes, ver_sep_boxes = pickle.load(open(path + '/debug.pkl','rb'))
    print(table_boxes)
    cell_cut(image, image, text_scale, table_boxes, hor_sep_boxes, ver_sep_boxes, use_advance_cut=False)
