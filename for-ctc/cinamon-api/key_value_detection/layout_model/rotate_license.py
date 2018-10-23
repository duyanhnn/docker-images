import cv2
import numpy as np
import glob
from skimage.morphology import skeletonize
from scipy.ndimage.morphology import binary_dilation, binary_erosion
from scipy.spatial.distance import cdist
import os
from math import sqrt
from .util import resize_to_prefered_height, resize_to_prefered_width
from .morph import compute_separators_morph_horizontal, detect_small_table, read_check_mark, getHoughLinesP, getContours
from .utils.image_utils.test_get_cell_background import merge_classify_license

def skelet(img, thres = 150, expand = False, dilation_iter = 2):
    img = img > thres
    img = skeletonize(img)
    img = binary_dilation(img, iterations=dilation_iter)

    if expand:
        pad = 5
        kernel_shape = (1, pad)
        kernel = np.ones(kernel_shape, dtype='uint8')
        img = binary_dilation(img, kernel, iterations=1)

    return (img * 255).astype('uint8')


from collections import defaultdict

def segment_by_angle_kmeans(lines, k=2, **kwargs):
    """Groups lines based on angle with k-means.

    Uses k-means on the coordinates of the angle on the unit circle
    to segment `k` angles inside `lines`.
    """

    # Define criteria = (type, max_iter, epsilon)
    default_criteria_type = cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER
    criteria = kwargs.get('criteria', (default_criteria_type, 10, 1.0))
    flags = kwargs.get('flags', cv2.KMEANS_RANDOM_CENTERS)
    attempts = kwargs.get('attempts', 10)

    # returns angles in [0, pi] in radians
    angles = np.array([line[1] for line in lines])
    # multiply the angles by two and find coordinates of that angle
    pts = np.array([[np.cos(2 * angle), np.sin(2 * angle)]
                    for angle in angles], dtype=np.float32)

    # run kmeans on the coords
    labels, centers = cv2.kmeans(pts, k, None, criteria, attempts, flags)[1:]
    labels = labels.reshape(-1)  # transpose to row vec

    # segment lines based on their kmeans label
    segmented_line = defaultdict(list)
    segmented_angle = defaultdict(list)
    for i, line in enumerate(lines):
        segmented_line[labels[i]].append(line[0])
        segmented_angle[labels[i]].append(line[1])

    segmented_line = list(segmented_line.values())
    segmented_angle = list(segmented_angle.values())

    angle_average = [ np.average([abs(np.cos(a)) for a in angles]) for angles in segmented_angle]
    # print(segmented_angle)
    # print(angle_average)
    indexes = np.argsort(angle_average)
    new_segmented = [segmented_line[i] for i in reversed(indexes)]

    return new_segmented


def split_seps(border_im):
    lines, debug_im = getHoughLinesP(border_im)
    hor_sep = np.zeros(border_im.shape[:2], dtype='uint8')
    ver_sep = np.zeros(border_im.shape[:2], dtype='uint8')
    combine_sep = np.zeros((border_im.shape[0],border_im.shape[1],3), dtype='uint8')
    lines_with_angle = []

    for i in range(len(lines)):
        width = 2
        x1, y1, x2, y2 = lines[i]

        slope = 1.0 * (y2 - y1) / (x2 - x1) if (x2 - x1) else np.pi / 2
        angle = np.arctan(slope)
        lines_with_angle.append((lines[i], angle))

    segmented = segment_by_angle_kmeans(lines_with_angle)

    for x1, y1, x2, y2 in segmented[0]:
        cv2.line(hor_sep, (x1, y1), (x2, y2), 255, width)

    for x1, y1, x2, y2 in segmented[1]:
        cv2.line(ver_sep, (x1, y1), (x2, y2), 255, width)

    kernel = np.ones((5, 5))
    hor_sep = cv2.morphologyEx(hor_sep, cv2.MORPH_CLOSE, kernel)
    ver_sep = cv2.morphologyEx(ver_sep, cv2.MORPH_CLOSE, kernel)

    combine_sep[hor_sep > 0] = [0, 255, 0]
    combine_sep[ver_sep > 0] = [0, 0, 255]

    return hor_sep, ver_sep, combine_sep

def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = int(det(d, xdiff) / div)
    y = int(det(d, ydiff) / div)
    return x, y

def dist(p1, p2):
    return sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def find_contour_lines(img, hor=True):
    img = ((img > 200) * 255).astype('uint8')
    rows, cols = img.shape[:2]
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((3,3)))
    size = min(img.shape[:2])
    debug_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    contours = getContours(img)
    rects = []

    if hor:
        y_pos_arr = []
    else:
        x_pos_arr = []

    filtered_cnts = []
    centers_arr = []
    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        (cx, cy), (w, h), angle = rect
        rects.append(rect)
        centers_arr.append((cx,cy))
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        length = max(w,h)
        if not length > size * 0.1: continue
        cv2.drawContours(debug_img, [box], 0, (0, 0, 255), 2)
        if hor and length > img.shape[1] * 0.1:
            y_pos_arr.append(cy)
        if not hor and length > img.shape[0] * 0.1:
            x_pos_arr.append(cx)

        filtered_cnts.append((cnt, (int(cx), int(cy)), length))

    avg_center = np.average(centers_arr, axis=0)

    length_arr = sorted([l for _, _, l in filtered_cnts])
    max_length = length_arr[-1]
    if hor:
        y_pos_arr = sorted(y_pos_arr)
        max_diff = y_pos_arr[-1] - y_pos_arr[0]
    else:
        x_pos_arr = sorted(x_pos_arr)
        max_diff = x_pos_arr[-1] - x_pos_arr[0]

    mult = 0.33 if length_arr[-3] > 0.8 * length_arr[-1] else 0.4
    length_thres = np.average(length_arr[-3:]) * mult
    print(length_thres)
    filtered_cnts = [(cnt, center, length) for cnt, center, length in filtered_cnts if length >= length_thres]
    print(len(filtered_cnts))

    # if hor:
    #     indexes = np.argsort([center[1] for _,center,_ in filtered_cnts])
    # else:
    #     indexes = np.argsort([center[0] for _, center, _ in filtered_cnts])

    converted_lines = []
    for cnt, _, _ in filtered_cnts:
        [vx, vy, x, y] = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
        if hor:
            lefty = int((-x * vy / vx) + y)
            righty = int(((cols - x) * vy / vx) + y)
            converted_lines.append(((cols - 1, righty), (0, lefty)))
        else:
            topx = int((-y * vx / vy) + x)
            bottomx = int(((rows - y) * vx / vy) + x)
            converted_lines.append(((topx, 0), (bottomx, rows - 1)))

    dist_arr = [line[0][1] if hor else line[0][0] for line in converted_lines]
    indexes = np.argsort(dist_arr)

    line_center_pos = [filtered_cnts[indexes[i]][1] if filtered_cnts[indexes[i]][2] > max_length * 0.75 else (0, 0) for i in [0, -1]]

    border_lines = []
    for line in (converted_lines[indexes[0]], converted_lines[indexes[-1]]):
        cv2.line(debug_img, line[0], line[1], (0, 255, 0), 2)
        border_lines.append(line)

    cv2.circle(debug_img, (int(avg_center[0]), int(avg_center[1])), 5, (255, 0, 0), -1)

    return border_lines, max_length, max_diff, line_center_pos, debug_img

def remap_points(points, origin_size, target_size):
    new_points = []
    oh, ow = origin_size
    nh, nw = target_size

    for p in points:
        new_p = int(1.0 * nw * p[0] / ow), int(1.0 * nh * p[1] / oh)
        new_points.append(new_p)

    return new_points

def order_points(pts):
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
    D = cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    (br, tr) = rightMost[np.argsort(D)[::-1], :]

    # return the coordinates in top-left, top-right,
    # bottom-right, and bottom-left order
    return np.array([tl, tr, br, bl], dtype="float32")

def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    #print("Rect: ", rect)
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # # check if area marked by 4 point is larger than 85% of the original image, then skip
    # h, w = image.shape[:2]
    # if 1.0 * maxWidth * maxHeight / (w * h) > 0.91 and use_face_to_rotate:
    #     print('Resizing image...')
    #     image = cv2.resize(image, (1200, 780))
    #
    #     return image

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    h, w = warped.shape[:2]

    # rotate the image counter-clockwise if h > w
    if w < h:
        warped = cv2.transpose(warped)
        cv2.flip(warped, 0, warped)

    return warped

def expand_line(root,target, length):
    x1, y1 = root
    x2, y2 = target
    lenAB = dist(root, target)
    new_x2 = int(x2 + (x2 - x1) / lenAB * length)
    new_y2 = int(y2 + (y2 - y1) / lenAB * length)

    return root, (new_x2, new_y2)

from numpy.linalg import norm

def dist_point2line(P, line):
    """ segment line AB, point P, where each one is an array([x, y]) """
    P = np.array(P)
    A, B = np.array(line[0]), np.array(line[1])

    d = (P[0]-A[0]) * (B[1]-A[1]) -(P[1]-A[1]) * (B[0]-A[0])

    n = B - A
    n = 1.0 / norm(n) * n
    P_ = A + n * np.dot(P - A, n)
    sign = 1 if P_[0] > P[0] else -1

    return sign * norm(np.cross(A-B, A-P))/norm(B-A)


WARPED_SIZE = (900, 520)

def rotate_license(border_im, debug_im, origin_im_normal_size,  filename = None, output_size = (1600, 925)):
    h, w = border_im.shape
    hor_sep, ver_sep, combine_sep = split_seps(border_im)
    hor_lines, max_hor_length, max_diff_y, center_pos_hor, hor_sep = find_contour_lines(hor_sep, hor=True)
    ver_lines, max_ver_length, max_diff_x, center_pos_ver, ver_sep = find_contour_lines(ver_sep, hor=False)

    points = []
    for l in hor_lines + ver_lines:
        cv2.line(debug_im, l[0], l[1], (0, 255, 0), 2)

    for hor_line in hor_lines:
        for ver_line in ver_lines:
            points.append(line_intersection(hor_line, ver_line))

    FALLBACK_OUTPUT = origin_im_normal_size, border_im, [], [], np.zeros(origin_im_normal_size.shape[:2], dtype='uint8'), [debug_im, debug_im]

    # safe guard for case with no borders detected
    if len(hor_line + ver_line) < 4:
        return FALLBACK_OUTPUT

    max_dist_hor = max(dist(points[0], points[1]), dist(points[2], points[3]))
    max_dist_ver = max(dist(points[0], points[2]), dist(points[1], points[3]))
    max_diff_x = max(max_hor_length, max_diff_x)
    max_diff_y = max(max_ver_length, max_diff_y)

    might_fail = (max_dist_hor < max_diff_x * 0.7) \
                 or (max_dist_ver < max_diff_y * 0.7)

    if max_hor_length > max_ver_length:
        hor_diff_thres = max_hor_length * 0.25
        for p1, p2, k in [(0,1,0), (2,3,1)]:
            if sum(center_pos_hor[k]) == 0: continue
            dist_p1_center = dist(points[p1], center_pos_hor[k])
            dist_p2_center = dist(points[p2], center_pos_hor[k])

            if dist_p1_center < hor_diff_thres:
                _, points[p1] = expand_line(points[p2], center_pos_hor[k], dist_p2_center)
            elif dist(points[p2], center_pos_hor[k]) < hor_diff_thres:
                _, points[p2] = expand_line(points[p1], center_pos_hor[k], dist_p1_center)

        cv2.circle(debug_im, center_pos_hor[0], 5, (255, 0, 0), -1)
        cv2.circle(debug_im, center_pos_hor[1], 5, (255, 0, 0), -1)

    else:
        ver_diff_thres = max_ver_length * 0.25
        for p1, p2, k in [(0, 2, 0), (1, 3, 1)]:
            if sum(center_pos_ver[k]) == 0: continue
            dist_p1_center = dist(points[p1], center_pos_ver[k])
            dist_p2_center = dist(points[p2], center_pos_ver[k])

            if dist_p1_center < ver_diff_thres:
                _, points[p1] = expand_line(points[p2], center_pos_ver[k], dist_p2_center)
            elif dist(points[p2], center_pos_ver[k]) < ver_diff_thres:
                _, points[p2] = expand_line(points[p1], center_pos_ver[k], dist_p1_center)

        cv2.circle(debug_im, center_pos_ver[0], 5, (255, 0, 0), -1)
        cv2.circle(debug_im, center_pos_ver[1], 5, (255, 0, 0), -1)

    for p in points:
        cv2.circle(debug_im, p, 5, (0, 0, 255), -1)

    # print(max_dist_hor, max_dist_ver, max_hor_length, max_ver_length)
    print('Fail test', might_fail, max_diff_x, max_diff_y)
    if might_fail and max_diff_y > 0.7 * h and max_diff_x > 0.6 * w:
        # skip de-warping
        return FALLBACK_OUTPUT

    border_warped = four_point_transform(border_im, np.array(points))
    border_warped = cv2.resize(border_warped, WARPED_SIZE)
    border_warped = skelet(border_warped, dilation_iter=1)
    # _, _, border_warped = find_contours(border_warped, hor=True)
    hor_sep_warped = compute_separators_morph_horizontal(border_warped, 100)

    mid = hor_sep_warped.shape[0] // 2

    print("dewarp")
    origin_warped = four_point_transform(origin_im_normal_size, np.array(
        remap_points(points, border_im.shape[:2], origin_im_normal_size.shape[:2])))

    if np.sum(hor_sep_warped[:mid,:]) < np.sum(hor_sep_warped[mid:,:]):
        cv2.flip(origin_warped, -1, origin_warped)
        cv2.flip(border_warped, -1, border_warped)

    origin_warped_out = cv2.resize(origin_warped, output_size)
    origin_warped = cv2.resize(origin_warped, WARPED_SIZE)
    debug_warped = origin_warped.copy()

    x1, y1, x2, y2 = 0, 0, origin_warped.shape[1], origin_warped.shape[0]
    content_mask = np.zeros((origin_warped.shape[:2]),dtype='uint8')
    face_x = int(x1 + (x2 - x1) * 0.64)
    face_y = int(y1 + (y2 - y1) * 0.25)

    content_mask[face_y:,face_x:] = 255
    # cv2.imwrite('mask.png', content_mask)

    small_table_x = int(x1 + (x2 - x1) * 0.345)
    small_table_y = int(y1 + (y2 - y1) * 0.75)
    small_table_pad = 20

    small_table_rect = (small_table_x - small_table_pad, small_table_y - small_table_pad), (int(0.52 * small_table_x + 0.48 * x2), y2 - 8)
    small_table_cells = detect_small_table(border_warped[small_table_rect[0][1]:small_table_rect[1][1],
                                           small_table_rect[0][0]:small_table_rect[1][0]], 30)
    is_cell_checked = []
    cell_ims = []
    for cell in small_table_cells:
        cell[0] += small_table_x - small_table_pad
        cell[2] += small_table_x - small_table_pad
        cell[1] += small_table_y - small_table_pad
        cell[3] += small_table_y - small_table_pad
        cell_im = origin_warped[cell[1]:cell[3], cell[0]:cell[2]]
        is_checked, cell_im_bin = read_check_mark(cell_im)
        is_cell_checked.append(is_checked)
        cell_ims.append(cell_im)
        cv2.rectangle(debug_warped, (cell[0]+1, cell[1]+1), (cell[2]-1, cell[3]-1), (0,255,0), 2)

    output_cell_checked = []
    output_cell_ims= []
    for i, cell in enumerate(small_table_cells):
        if i not in [0,8]:
            output_cell_checked.append(is_cell_checked[i])
            output_cell_ims.append(cell_ims[i])
            if is_cell_checked[i]:
                cv2.rectangle(debug_warped, (cell[0]+1, cell[1]+1), (cell[2]-1, cell[3]-1), (0,0,255), 2)

    # cv2.rectangle(debug_warped, small_table_rect[0], small_table_rect[1], (255,0,0), 2)
    cv2.rectangle(debug_warped, (face_x, face_y), (x2, y2), (255, 0, 0), 2)

    if filename is not None:
        debug_dir = os.path.dirname(filename)
        basename = os.path.basename(filename).split('.')[0]
        cv2.imwrite(os.path.join(debug_dir, "hor.png"), hor_sep)
        cv2.imwrite(os.path.join(debug_dir, "ver.png"), ver_sep)
        cv2.imwrite(os.path.join(debug_dir, "combined.png"), combine_sep)
        cv2.imwrite(os.path.join(debug_dir, "{}_debug_im.png".format(basename)), debug_im)
        cv2.imwrite(os.path.join(debug_dir, "{}_warped.png".format(basename)), origin_warped_out)
        # cv2.imwrite(os.path.join(debug_dir, "{}_border_warped.png".format(basename)), border_warped)

    return origin_warped_out, border_warped, output_cell_checked, output_cell_ims, content_mask, (debug_warped, debug_im)


def read_color_license(lines, im):
    for l in lines:
        x1, y1, x2, y2 = l
        h = y2 - y1
        if h > 40 and x1 < 50:
            region_im = im[y1:y2, x1:x2]
            color_id = merge_classify_license(region_im)
            color_map = {0: 'gold', 1: 'blue', 2: 'green'}
            print('License color: {}'.format(color_map[color_id]))
            return color_id, (x2, y2)

    return -1 , im.shape[:2]


if __name__ == '__main__':
    for filename in glob.glob("/home/taprosoft/Downloads/test_segmented/flax_bprost/run/data_zip/license_new/test/license/*_GT0.jpg"):

        print('Processing ' + filename)
        img = cv2.imread(filename)

        origin_im_base = filename[:-8]
        origin_im = cv2.imread(origin_im_base + '.jpg')
        origin_im_normal_size = origin_im.copy()
        h, w = origin_im.shape[:2]

        if h > w * 1.3:
            origin_im = resize_to_prefered_width(origin_im, 1200)  #2400
        else:
            origin_im = resize_to_prefered_height(origin_im, 800) #2200

        print(origin_im.shape)

        origin_im_copy = origin_im.copy()
        # cv2.imwrite(origin_im_base + '.jpg', origin_im_copy)

        h, w = origin_im.shape[:2]

        print('org shape: {}'.format(origin_im.shape[:2]))

        for i in [0]: #reversed(range(0, 3)):
            gt0 = cv2.imread(origin_im_base + '_GT{}.jpg'.format(i), 0)
            # gt1 = cv2.imread(origin_im_base + '_GT1.jpg', 0)

            print('GT{} shape: {}'.format(i, gt0.shape[:2]))
            # print('GT1 shape: {}'.format(gt1.shape[:2]))

            gt0 = cv2.resize(gt0, (w, h))
            # gt1 = cv2.resize(gt1, (w, h))

            if i == 0:
                gt0 = skelet(gt0, thres=100, expand=False)
                # gt0 = np.roll(gt0, -3, axis=1)
            elif i == 1:
                gt0 = skelet(gt0, thres=80)
            elif i == 2:
                gt0 = ((gt0 > 150) * 255).astype('uint8')

            if i == 0:
                origin_im[gt0 > 100] = [0, 255, 0]
            elif i == 1:
                origin_im[gt0 > 100] = [255, 0, 0]
                pass
            elif i == 2:
                origin_im[gt0 > 150] = [0, 0, 255]

            # cv2.imwrite(filename[:-4] + "_out.png", (img * 255).astype('uint8'))
            # cv2.imwrite(filename, (img * 255).astype('uint8'))

            # if i in [0,1,2]:
            #     cv2.imwrite(origin_im_base + '_GT{}.jpg'.format(i), gt0)

            origin_warped, border_warped, _, _, debug_im = rotate_license(gt0, origin_im, origin_im_normal_size, filename)

            # cv2.imwrite(origin_im_base + '_GT1.jpg', gt1)
        debug_com = debug_im[0].copy()
        debug_com[border_warped > 0] = [0, 255, 0]
        debug_im = cv2.addWeighted(debug_im[0], 0.5, debug_com, 0.5, 0)
        cv2.imwrite(origin_im_base + '_com.png', debug_im)