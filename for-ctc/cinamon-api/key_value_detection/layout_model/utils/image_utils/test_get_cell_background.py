from __future__ import print_function
from __future__ import division

from .test_cell_binarize import (
    np, cv2, sys,
    combine_favor_black, simple_otsu,
    ocropus_binarize_code, plt, plot_gray,
    local_imutils, get_x_y_start_stop_contour,
    get_bbox_x_y_w_h, fill_box_bin_img,
    get_x_y_start_stop, get_bboxs_max_margin,
    get_cell_merging_pix_gray
)


def get_cell_merging_2_step(cell, background, merging_thresh=40,
                            white_merge_thresh=15, margin=None,
                            debug=False):
    if background > 200:  # Leave it be
        cell_merging = get_cell_merging_pix_gray(
            cell, background, merging_thresh)
        return cell_merging

    cell_merging = get_cell_merging_pix_gray(cell, background, merging_thresh)
    colors_non_background_cells = cell[np.where(1 - cell_merging)]
    # colors_background_cells = cell[cell_merging]

    if len(colors_non_background_cells) > 0:
        mean = np.mean(colors_non_background_cells)
    else:
        if debug:
            print("CellBinarization::Warning: Blank Image?")
        return cell_merging

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
    return cell_merging


def get_single_background_cell(colored_cell, cell_msers, debug=False):
    '''
    Input:
        colored_cell: 3 channel,
        cell_msers: mser masking from parent cell
    Output:
        cell_color: 1 value, 3 channel b,g,r
    '''
    cell_color = []

    if len(np.where(cell_msers == 0)[0]) == 0:
        cell_msers = np.zeros_like(cell_msers)
    for i in range(3):
        cell_color.append(np.median(colored_cell[:, :, i][cell_msers == 0]))
    return np.array(cell_color).astype(np.uint8)


def get_cell_background(colored_cell, margin=5, debug=False):
    """
    1. Use ocropus nlbin to binarize
    2. Apply mser
    3. Merge msers
    4. Find outer contours each group
    5. Find background each group
    6. binarize each group using adaptive threshold"""
    cell = cv2.cvtColor(colored_cell, cv2.COLOR_BGR2GRAY)
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

    character_maps = np.zeros((cell.shape[0], cell.shape[1])).astype('uint8')
    cv2.drawContours(character_maps, contours_ocadap, -1, 255, 3)
    # 3.
    # kernel = np.ones((5, 20))
    # closing = cv2.morphologyEx(character_maps, cv2.MORPH_CLOSE, kernel)
    closing = character_maps
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
    cell_background = np.zeros_like(colored_cell)

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
        child_cell = colored_cell[y0:y1, x0:x1].copy()
        child_msers_mask = closing[y0:y1, x0:x1].copy()
        child_cell_background = get_single_background_cell(
            child_cell, child_msers_mask)
        for i in range(3):
            cell_background[y0:y1, x0:x1, i] = child_cell_background[i]

    return cell_background


def get_single_background_colored_field_license(colored_cell):
    cell_background = get_cell_background(colored_cell)
    mask = cell_background[:, :, 0] == 0

    if len(np.where(mask == 0)[0]) == 0:
        mask = np.zeros_like(mask)
    color = get_single_background_cell(cell_background,
                                       cell_background[:, :, 0] == 0,
                                       debug=False)
    return color


def classify_for_license(color):
    '''
    output: classified - 0: brown, 1 blue, 2 green
    '''
    if np.argmax(color) == 0 and (color[0]/color[2]) > 1.3:
        return 1
    elif np.argmax(color) == 1 and\
            (color[1]/color[0] > 1.3) and (color[1]/color[2]) > 1.2:
        return 2
    if np.argmax(color) == 2:
        return 0
    # Br1: 63, 93, 110
    # Br2: 67, 112, 133
    BGR_BROWN = np.array([63, 93, 110]).astype('uint8')
    # Bl1: 231, 179, 102
    # Bl2: 224, 151, 71
    # Bl3: 221, 150, 76
    BGR_BLUE = np.array([231, 179, 102]).astype('uint8')
    # Gr1: 91, 202, 150
    BGR_GREEN = np.array([91, 202, 150]).astype('uint8')
    color_array = np.array([[BGR_BROWN, BGR_GREEN, BGR_BLUE,
                             color.astype('uint8')]]).astype('uint8')
    lab_color_array = cv2.cvtColor(color_array, cv2.COLOR_BGR2LAB)
    lab_samples = lab_color_array[0, :3, :2]
    lab_color = lab_color_array[:, 3, :2]

    dists = [np.dot(lab_color[0] - lab_samples[i],
                    lab_color[0] - lab_samples[i]) for i in range(3)]
    return np.argmin(dists)


def merge_classify_license(colored_cell):
    color = get_single_background_colored_field_license(colored_cell)
    classified = classify_for_license(color)
    return classified


if __name__ == '__main__':
    colored_cell = cv2.imread(sys.argv[1])
    cell_background = get_cell_background(colored_cell)
    color = get_single_background_colored_field_license(colored_cell)
    print("Color: ", color)
    classified = classify_for_license(color)
    for i in range(3):
        colored_cell[:, :, i] = color[i]
    print(classified)
    out_cell_name = 'out_cell_background.jpg'
    if len(sys.argv) >= 3:
        out_cell_name = sys.argv[2]
    cv2.imwrite(out_cell_name, colored_cell)
