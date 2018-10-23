from __future__ import division
import numpy as np
import cv2
from numpy import zeros
from ..local_imutils import (connected_component_analysis_sk as cca_sk,
                             ocropy_binarize_grayscale, invert_color_image)
from ..bboxs_tools.bbox_operations import (
    fill_box_bin_img, get_width_height_bounding_boxs,
    get_boundingbox_tl_lr, crop_bbox)
from ..binarize_algos.ocropus_binarize import ocropus_binarize_code


class X_Y_TreeNode(object):
    def __init__(self):
        self.childs = []
        self.x_offset = 0
        self.y_offset = 0
        self.width = 0
        self.height = 0

    def draw(self, img, color):
        if len(self.childs) == 0:
            cv2.rectangle(img, (self.x_offset, self.y_offset),
                          (self.x_offset+self.width,
                          self.y_offset+self.height),
                          color, 2)
        for child in self.childs:
            child.draw(img, color)

    def extract_childs(self, img):
        childs = []
        if len(self.childs) == 0:
            childs.append(img[
                self.y_offset:self.y_offset+self.height,
                self.x_offset:self.x_offset+self.width])
        for child in self.childs:
            childs.extend(child.extract_childs(img))
        return childs

    def extract_childs_coords(self):
        childs = []
        if len(self.childs) == 0:
            childs.append(
                (self.x_offset, self.y_offset,
                 self.x_offset+self.width, self.x_offset+self.height))
        for child in self.childs:
            childs.extend(child.extract_childs_coords())
        return childs



def get_first_x_start_x_stop_proj_cut(projections, thres=5):
    i = 0
    x_start_rel = 0
    x_stop_rel = 0
    while(i < projections.shape[0] and projections[i] >= thres):
        i += 1
    x_start_rel = i
    x_stop_rel = i
    while(i < projections.shape[0] and projections[i] < thres):
        i += 1
    x_stop_rel = i
    return x_start_rel, x_stop_rel


def cut_projection(projection, thres_width, thres):
    x_start = 0
    x_stop = projection.shape[0]
    cursor = 0

    x_start_next = x_start
    groups = []
    while(cursor < x_stop-x_start):
        x_start_rel, x_stop_rel = get_first_x_start_x_stop_proj_cut(
            projection[cursor:], thres)
        if x_stop_rel-x_start_rel >= thres_width:
            groups.append((x_start_next+x_start_rel,
                           x_start_next+x_stop_rel))
        x_start_next = x_start_next+x_stop_rel
        cursor = cursor + x_stop_rel

    final_groups = []
    if len(groups) > 1:
        final_groups = [(0, groups[0][0])]
        for i in range(len(groups)):
            if i < len(groups) - 1:
                final_groups.append((groups[i][1], groups[i+1][0]))
        final_groups.append((groups[-1][1], projection.shape[0]))
    return final_groups, groups


def divide_groups_i_max_space(white_space_groups, i_max_space, max_shape):
    """
    Args:
        white_space_groups: groups of white_spaces
        i_max_space: index of maximum space
        max_shape: boundary
    """
    groups = []
    if i_max_space > 0:
        for i in range(i_max_space):
            groups.append((white_space_groups[i][1],
                           white_space_groups[i+1][0]))
    else:
        groups = [(0, white_space_groups[i_max_space][0])]
    if i_max_space < len(white_space_groups) - 1:
        groups.append(
            (white_space_groups[i_max_space][1],
             white_space_groups[i_max_space+1][0]))
    else:
        groups.append(
            [white_space_groups[i_max_space][1], max_shape])
    return groups


def xy_cut(array_binimg, thres_ratio_x=1, thres_ratio_y=0.7, x_offset=0,
           y_offset=0, is_secondtime=False, avg_width=None, avg_height=None):
    # Create root node
    root = X_Y_TreeNode()
    root.width, root.height = array_binimg.shape[1], array_binimg.shape[0]
    root.x_offset, root.y_offset = x_offset, y_offset
    if not is_secondtime:
        bounding_rects = cca_sk(array_binimg)
        if bounding_rects is None:
            return root

        widths, heights = get_width_height_bounding_boxs(bounding_rects)

        avg_width = sum(widths)/len(widths)
        avg_height = sum(heights)/len(heights)
        thres_v, thres_h = avg_width*thres_ratio_x, avg_height*thres_ratio_y
        # makes a blank canvas to draw bounding rects
        empty_img = zeros(array_binimg.shape)
        # Fill all the rects
        for bounding_rect in bounding_rects:
            empty_img = fill_box_bin_img(empty_img, bounding_rect, 1)
    else:
        empty_img = array_binimg
        thres_v, thres_h = avg_width*thres_ratio_x, avg_height*thres_ratio_y
    # Calculate projection Horizontal, Vertical
    h_project, v_project = np.sum(empty_img, axis=1), np.sum(empty_img, axis=0)

    y_groups, white_ys = cut_projection(h_project, thres_h, 3)
    x_groups, white_xs = cut_projection(v_project, thres_v, 3)

    if len(x_groups) <= 1 and len(y_groups) <= 1:
        return root
    if len(white_ys) == 0:
        y_groups = [(0, array_binimg.shape[0])]
    elif len(white_xs) == 0:
        x_groups = [(0, array_binimg.shape[1])]
    elif len(white_ys) >= 1 and len(white_xs) >= 1:
        widths = [group[1]-group[0] for group in white_xs]
        heights = [group[1]-group[0] for group in white_ys]
        max_width = max(widths)
        i_max_width = widths.index(max_width)
        max_height = max(heights)
        i_max_height = heights.index(max_height)
        if max_width > max_height:
            x_groups = divide_groups_i_max_space(white_xs, i_max_width,
                                                 array_binimg.shape[1])
            y_groups = [(0, array_binimg.shape[0])]
        else:
            y_groups = divide_groups_i_max_space(white_ys, i_max_height,
                                                 array_binimg.shape[0])
            x_groups = [(0, array_binimg.shape[1])]

    for x_group in x_groups:
        for y_group in y_groups:
            if np.sum(empty_img[y_group[0]+1:y_group[1],
                                x_group[0]+1:x_group[1]]) == 0:
                print("Warning: canceling box since no cc found")
                continue
            bbox = get_boundingbox_tl_lr([y_group[0], x_group[0]],
                                         [y_group[1], x_group[1]])
            new_image = crop_bbox(array_binimg, bbox)
            new_child = xy_cut(new_image, min(thres_ratio_x*1, 1.5),
                               min(thres_ratio_y*1, 3),
                               root.x_offset+x_group[0],
                               root.y_offset+y_group[0],
                               True,
                               avg_width,
                               avg_height)
            if new_child is not None:
                root.childs.append(new_child)
    return root


def fullfile_xy_cut(img_path, out_path):
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
    img_bin = ocropus_binarize_code(img)
    img_bin = invert_color_image(img_bin)
    tree_root = xy_cut(img_bin)
    img = cv2.imread(img_path)
    tree_root.draw(img, (0, 0, 255))
    cv2.imwrite(out_path, img)
