from __future__ import print_function
from __future__ import division
import cv2
import numpy as np
from skimage.filters import threshold_niblack, rank
from skimage.morphology import disk
import sys
from test_cell_binarize import (
    cell_binarize_mser, simple_otsu, simple_adaptive, ocropus_binarize_code,
    denoising_favor_white, plt, plot_gray, local_imutils, clahe_image,
    combine_favor_black)

def variance_of_laplacian(image):
    # compute the Laplacian of the image and then return the focus
    # measure, which is simply the variance of the Laplacian
    return cv2.Laplacian(image, cv2.CV_64F).var()


def nick_binarize(img):
    '''Binarize linecut images using two differently sized local threshold kernels

    Args:
    img_list: list of grayscale linecut images
    Returns:
    results: binarized images in the same order as the input'''

    assert img.ndim == 2, 'Image must be grayscale'

    # Resize the images to 200 pixel height
    scaling_factor = 200/img.shape[0]
    new_w = int(scaling_factor*img.shape[1])
    new_h = int(scaling_factor*img.shape[0])
    img = cv2.resize(img, (new_w, new_h))

    # First pass thresholding
    th1 = threshold_niblack(img, 31, 0.0)

    # Second pass thresholding
    radius = 201
    structured_elem = disk(radius)
    th2 = rank.otsu(img, structured_elem)

    # Masking
    img = (img > th1) | (img > th2)
    img = img.astype('uint8')*255

    return img


def nick_binarize_modified(img):
    '''Binarize linecut images using two differently sized local threshold kernels

    Args:
    img_list: list of grayscale linecut images
    Returns:
    results: binarized images in the same order as the input'''

    assert img.ndim == 2, 'Image must be grayscale'

    if img.shape[0] >= 200:
        return cv2.resize(nick_binarize(img), (img.shape[1], img.shape[0]))
    # Resize the images to 200 pixel height
    scaling_factor = 200.0/img.shape[0]
    if scaling_factor > 15:     # Too small, the whole image is mask
        return np.zeros_like(img).astype(np.uint8)

    window_size = int(31/scaling_factor)
    window_size = window_size + window_size % 2 + 1
    # First pass thresholding
    th1 = threshold_niblack(img, window_size, 0.0)

    # Second pass thresholding
    radius = int(201/scaling_factor)
    radius = radius + 1 - radius % 2
    structured_elem = disk(radius)
    th2 = rank.otsu(img, structured_elem)

    # Masking
    img = (img > th1) | (img > th2)
    img = img.astype('uint8')*255

    return img


def single_cell_binarize_with_background_nick(
        cell, background, merging_thresh=40,
        white_merge_thresh=15, margin=None,
        debug=False):
    customized_adaptive = simple_adaptive
    customized_otsu = simple_otsu
    if debug:
        print("estimated background color: ", background)
    if background > 200:            # Leave it be
        return ocropus_binarize_code(cell), denoising_favor_white(cell)

    cell_merging = np.abs(cell - background) < merging_thresh*np.max(cell)/255
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

    '''
    new_cell_nick = cv2.resize(
        nick_binarize(new_cell), (new_cell.shape[1], new_cell.shape[0]))
    '''
    new_cell_nick = nick_binarize_modified(new_cell)
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

    new_cell = (255*np.logical_or(new_cell, new_cell_nick).astype('uint8'))
    '''
    new_cell = (new_cell.astype('uint8')/1.5).astype('uint8')
    ret2, new_cell = cv2.threshold(new_cell, 0,
                                   255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    new_cell = hvs_binarize(new_cell)
    '''
    return cell, new_cell.astype(np.uint8)


if __name__ == '__main__':
    img = cv2.imread(sys.argv[1])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img, thresh = cell_binarize_mser(
        img,
        customized_method=single_cell_binarize_with_background_nick,
        debug=False
        )
    out_path = 'out_cell_cert.jpg'
    if len(sys.argv) >= 3:
        out_path = sys.argv[2]
    cv2.imwrite(out_path, thresh)
