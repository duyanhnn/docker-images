from __future__ import print_function
from __future__ import division

import numpy as np
from PIL import Image
import cv2
import sys
from . import local_imutils
from .test_clahe import clahe_image
from .bboxs_tools.bbox_operations import (
    get_x_y_start_stop_contour, get_bbox_x_y_w_h,
    get_x_y_start_stop, fill_box_bin_img
)

from skimage.filters import threshold_otsu, threshold_local
from .binarize_algos.ocropus_binarize import (
    ocropus_binarize_code
)
from ..logger_utils import plot_gray, plt

block_size = 35


def gaussian_otsu(img, ksize=3):
    ksize = int(max(max(img.shape), 1000)/200)
    if ksize % 2 == 0:
        ksize -= 1
    blur = cv2.GaussianBlur(img, (ksize, ksize), 0)
    return simple_otsu(blur)


def gaussian_adaptive(img, ksize=3, block_size=35):
    ksize = int(max(max(img.shape), 1000)/200)
    if ksize % 2 == 0:
        ksize -= 1
    blur = cv2.GaussianBlur(img, (ksize, ksize), 0)
    return simple_adaptive(blur)


def simple_adaptive(img, block_size=35):
    blur = img
    try:
        sk_thresh = threshold_local(blur, block_size, offset=10)
    except ValueError:
        print("Warning::CELL_BINARIZE::STEP1: blank image?")
        sk_thresh = 0
    return (255*(blur > sk_thresh)).astype(
            np.uint8)


def simple_otsu(img):
    blur = img
    try:
        sk_thresh = threshold_otsu(blur)
    except ValueError:
        print("Warning::CELL_BINARIZE::STEP1: blank image?")
        sk_thresh = 0
    return (255*(blur > sk_thresh)).astype(
            np.uint8)


def tri_step_binarize(img, first_binarize=simple_otsu,
                      second_binarize=simple_adaptive,
                      third_binarize=ocropus_binarize_code):
    bin1 = first_binarize(img)/255
    bin2 = second_binarize(img)/255
    bin3 = ocropus_binarize_code(img)/255
    bin = bin1 + bin2 + bin3
    bin = 255*(bin >= 2.0).astype('uint8')
    return bin


def combine_favor_white(img, first_method=simple_otsu,
                        second_method=simple_adaptive):
    return 255*(np.logical_or(first_method(img), second_method(img))
                ).astype('uint8')


def combine_favor_black(img, first_method=gaussian_adaptive,
                        second_method=gaussian_otsu):
    return 255*(np.logical_and(first_method(img), second_method(img))
                ).astype('uint8')


def denoising_favor_white(img, first_method=simple_otsu,
                          second_method=simple_adaptive,
                          denoise_func=gaussian_otsu):
    img_normal = combine_favor_white(img, first_method, second_method)
    # img_normal = tri_step_binarize(img)
    img_denoised = denoise_func(img)
    return (255*np.logical_or(img_normal, img_denoised)).astype('uint8')


def get_background_color_cell(cell, cell_merging):
    if len(cell.shape) < 3:
        return np.median(cell)
    return np.array([np.median(cell[:, :, 0][cell_merging]),
                     np.median(cell[:, :, 1][cell_merging]),
                     np.median(cell[:, :, 2][cell_merging])])


def get_cell_merging_pix_gray(cell, background, merging_thresh):
    '''
    cell: grayscale
    backgroun: estimated background
    merging_thresh: close to background
    '''
    return np.abs(cell - background) < merging_thresh*np.max(cell)/255





def single_cell_binarize_with_background(cell, background, merging_thresh=40,
                                         white_merge_thresh=15, margin=None,
                                         debug=False):
    customized_adaptive = simple_adaptive
    customized_otsu = simple_otsu
    if debug:
        print("estimated background color: ", background)
    if background > 200:            # Leave it be
        return ocropus_binarize_code(cell), denoising_favor_white(cell)

    cell_merging = get_cell_merging_pix_gray(cell, background, merging_thresh)
    colors_non_background_cells = cell[np.where(1 - cell_merging)]
    # colors_background_cells = cell[cell_merging]

    if len(colors_non_background_cells) > 0:
        mean = np.mean(colors_non_background_cells)
    else:
        if debug:
            print("CellBinarization::Warning: Blank Image?")
        return cell, cell

    if debug:
        print("estimated foreground: ", mean)
    if mean > background:
        cell = 255 - cell
        background = 255 - background

    if debug:
        plt.figure()
        plt.subplot(2, 1, 1)
        plot_gray(cell, plt_title="Cell after optional inverting step")
    cell_merging = cell >= background
    non_background_cells = 1 - cell_merging
    colors_non_background_cells = cell[np.where(non_background_cells)]

    new_cell = np.copy(cell)

    new_cell[np.where(cell_merging)] = 255
    cell = ocropus_binarize_code(cell, zoom=0.5)

    new_cell[np.where(non_background_cells)] = colors_non_background_cells

    # For use alone only
    if margin:
        new_cell_inv = 255 - new_cell
        labels, bboxs =\
            local_imutils.connected_component_analysis_sk_with_labels(
                new_cell_inv)
        unique_labels = sorted(list(np.unique(labels)))[1:]
        for i in range(len(non_background_cells)):
            y, x = non_background_cells[i]
            if abs(y) < margin or (cell.shape[0] - y < margin) or\
                    x < margin or (cell.shape[1] - x < margin):
                if labels[y, x] in unique_labels:
                    new_cell[labels == labels[y, x]] = 255
                    cell[labels == labels[y, x]] = 255
                unique_labels = [label for label in unique_labels
                                 if label != labels[y, x]]

    if debug:
        plt.subplot(2, 1, 2)
        plot_gray(new_cell, show_now=True, plt_title="New cell")
    new_cell = cv2.cvtColor(
        clahe_image(cv2.cvtColor(new_cell.astype('uint8'),
                                 cv2.COLOR_GRAY2BGR)).astype('uint8'),
        cv2.COLOR_BGR2GRAY)
    new_cell = denoising_favor_white(new_cell, customized_otsu,
                                     customized_adaptive, combine_favor_black)

    '''
    new_cell = (new_cell.astype('uint8')/1.5).astype('uint8')
    ret2, new_cell = cv2.threshold(new_cell, 0,
                                   255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    new_cell = hvs_binarize(new_cell)
    '''
    return cell, new_cell.astype(np.uint8)


def cell_binarize(cell, merging_thresh=40, white_merge_thresh=15, margin=10,
                  customized_method=single_cell_binarize_with_background,
                  debug=False):
    """Binarize Cutted Cell
    cell: grayscale image
    return: another grayscale image
    """
    '''
    cell = cv2.cvtColor(clahe_image(cv2.cvtColor(cell, cv2.COLOR_GRAY2BGR)),
                        cv2.COLOR_BGR2GRAY)
    '''
    # cell = ocropus_binarize_code(cell)
    # cell = ocropus_binarize_code(cell, zoom=0.5)
    background = np.median(cell)
    return customized_method(cell, background,
                             merging_thresh, white_merge_thresh,
                             margin, debug=debug)


def cell_binarize_mser(cell, margin=5,
                       customized_method=single_cell_binarize_with_background,
                       debug=False):
    """
    1. Use ocropus nlbin to binarize
    2. Apply mser
    3. Merge msers
    4. Find outer contours each group
    5. Find background each group
    6. binarize each group using adaptive threshold"""
    # 1.
    cell_binarized_ocadap = combine_favor_black(cell, simple_otsu,
                                                ocropus_binarize_code)
    # 2.
    mser_detector = cv2.MSER_create(
        _min_area=6,
        _max_area=int(cell.shape[0]*cell.shape[1]/4)
    )
    contours_ocadap, cv_bboxs = mser_detector.detectRegions(
        cell_binarized_ocadap)
    contours, cv_bboxs = mser_detector.detectRegions(cell)
    character_maps = np.zeros((cell.shape[0], cell.shape[1])).astype('uint8')
    cv2.drawContours(character_maps, contours_ocadap, -1, 255, 3)
    cv2.drawContours(character_maps, contours, -1, 255, 3)
    # 3.
    kernel = np.ones((5, 20))
    closing = cv2.morphologyEx(character_maps, cv2.MORPH_CLOSE, kernel)
    # closing = cv2.dilate(character_maps, kernel, iterations=2)
    # closing = cv2.morphologyEx(closing, cv2.MORPH_CLOSE, kernel)

    if debug:
        plt.figure()
        plt.subplot(3, 1, 1)
        plot_gray(cell_binarized_ocadap,
                  plt_title="merged from ocropus & adaptive thresh")
        plt.subplot(3, 1, 2)
        plot_gray(character_maps,
                  plt_title="Character detect by mser")
        plt.subplot(3, 1, 3)
        plot_gray(closing, plt_title="closing of contours")
        plt.show()

    # 4.
    labels, bboxs = local_imutils.connected_component_analysis_sk_with_labels(
        closing)
    # unique_labels = sorted(list(np.unique(labels)))[1:]
    # non_background_cells = np.where(character_maps == 255)

    im2, contours, hierarchy = cv2.findContours(closing.astype(np.uint8),
                                                cv2.RETR_TREE,
                                                cv2.CHAIN_APPROX_SIMPLE)
    # 5, 6
    # should try merging here
    # Contours [m, n, 1, 2]
    contours = np.array(contours)
    # cell_binarized = cell.copy()  # np.ones(character_maps.shape)*255
    cell_binarized = np.ones(character_maps.shape)*255

    new_cell = np.ones(character_maps.shape)*255
    character_maps_inv = local_imutils.invert_color_image(
        closing.astype('uint8'))
    bboxs = []
    # Merging cell
    for contour in contours:
        x0_0, y0_0, x1_0, y1_0 = get_x_y_start_stop_contour(contour)
        bboxs.append(get_bbox_x_y_w_h(x0_0, y0_0, x1_0-x0_0+1, y1_0-y0_0+1))

    text_regions = np.zeros_like(cell)
    for bbox in bboxs:
        text_regions = fill_box_bin_img(text_regions, bbox, 255)
    labels, bboxs = local_imutils.connected_component_analysis_sk_with_labels(
        text_regions)
    if debug:
        plot_gray(text_regions, new_figure=True, show_now=True)
    for bbox in bboxs:
        x0_0, x1_0, y0_0, y1_0 = get_x_y_start_stop(bbox)
        x0, y0, x1, y1, margin_new = get_bboxs_max_margin(
            cell, margin, x0_0, y0_0, x1_0, y1_0)

        if debug:
            print(x0, y0, x1, y1)
        child_cell = cell[y0:y1, x0:x1].copy()
        '''
        new_child_cell, child_cell_binarized =\
            single_cell_binarize_with_background(child_cell, background,
                                                 merging_thresh=2,
                                                 margin=None)
                                                 '''

        # keeping old margin
        new_child_cell, child_cell_binarized = cell_binarize(
            child_cell, merging_thresh=2, margin=None,
            customized_method=customized_method, debug=debug)
        cell_binarized[y0:y1, x0:x1] = 255*np.logical_and(
            cell_binarized[y0:y1, x0:x1], child_cell_binarized[:, :])
        if debug:
            plt.figure()
            plt.subplot(1, 3, 1)
            plot_gray(child_cell, "Before binaize")
            plt.subplot(1, 3, 2)
            plot_gray(child_cell_binarized, "After binarized")
            plt.subplot(1, 3, 3)
            plot_gray(cell_binarized, "Merge binarized cell")
            plt.show()

        # cell_binarized[y0:y1, x0:x1] = child_cell_binarized[:, :]

        # cell_binarized[y0:y1, x0:x1] = background
        new_cell[y0:y1, x0:x1] = new_child_cell[:, :]

    '''
    cell_binarized = 255*np.logical_or(
            cell_binarized,
            character_maps_inv
        ).astype(np.uint8)
    '''
    if debug:
        plt.figure()
        plt.subplot(1, 2, 1)
        plot_gray(character_maps_inv, "Character maps, invert")
        plt.subplot(1, 2, 2)
        plot_gray(cell_binarized, "Binarized Cell")
        plt.show()
    ''' # Unusable on 2+ backgrounds case
    cell_binarized = 255*np.logical_or(
            cell_binarized,
            255 - cell_binarized_ocadap
        ).astype(np.uint8)
    '''

    return new_cell, cell_binarized


def get_bboxs_max_margin(cell, margin, x0_0, y0_0, x1_0, y1_0):
    '''Adding margin, if out of range, return max bbox
    Args: cell, margin, x0_0, y0_0, x1_0, y1_0
    Return: new x0, y0, x1, y1, margin_new
    '''
    x0 = max(x0_0 - margin, 0)
    y0 = max(y0_0 - margin, 0)
    x1 = min(x1_0 + margin, cell.shape[1])
    y1 = min(y1_0 + margin, cell.shape[0])
    margin_new = min([x0_0 - x0, x1 - x1_0, y0_0 - y0, y1 - y1_0])
    x0 = x0_0 - margin_new
    y0 = y0_0 - margin_new
    x1 = x1_0 + margin_new
    y1 = y1_0 + margin_new
    return x0, y0, x1, y1, margin_new


if __name__ == "__main__":
    old_cell = np.array(Image.open(sys.argv[1]))
    # cell = np.array(Image.open(sys.argv[1]).convert('L'))
    cell = cv2.cvtColor(cv2.imread(sys.argv[1]), cv2.COLOR_BGR2GRAY)
    cell, new_cell = cell_binarize_mser(cell,
                                        debug=False)
    out_path = 'out_cell.jpg'
    if len(sys.argv) >= 3:
        out_path = sys.argv[2]
    cv2.imwrite(out_path, new_cell)
    # cv2.imwrite('out_cell_original.jpg', cell)
