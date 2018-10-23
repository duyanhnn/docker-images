from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
import cv2
import numpy as np
from ..bboxs_tools import bbox_operations as bbops
from ..local_imutils import (
    border_following_clockwise,
    connected_component_analysis_sk_with_labels as cca_sk)


class TreeMSER(object):
    def __init__(self, the_parent=None, the_area=0, the_contour=None,
                 the_bbox=None):
        """
        should take intensity into account
        """
        self.area = the_area
        self.contour = the_contour
        self.parent = the_parent
        self.bbox = the_bbox
        self.children = []
        self.val = 0
        self.level = 0
        self.sample_pixel = None
        # Make sure children area always sorted top down

    def update_contour(self, imggray):
        if self.sample_pixel is None:
            return
        cropped_bbox = bbops.crop_bbox(imggray, self.bbox)
        cropped_bbox = cropped_bbox > self.level
        labels, bboxs = cca_sk(cropped_bbox)
        x_start, _, y_start, _ = bbops.get_x_y_start_stop(self.bbox)
        x0 = self.sample_pixel[0] - x_start
        y0 = self.sample_pixel[1] - y_start
        label = labels[y0, x0]
        labels_mask = np.zeros_like(labels)
        labels_mask[labels == label] = 255
        """
        label = labels[y0, x0]
        points = np.where(labels == label)
        points = np.transpose(np.array([points[1], points[0]]))
        self.contour = border_following_clockwise(points)
        """
        print(labels_mask.shape)
        _, contours, _ = cv2.findContours(labels_mask.astype(np.uint8).copy(),
                                          cv2.RETR_TREE,
                                          cv2.CHAIN_APPROX_SIMPLE)
        """
        self.contour[:, 0] += x_start
        self.contour[:, 1] += y_start

        self.contour = np.reshape(self.contour, (self.contour.shape[0], 1, 2))
        """
        self.contour = contours[0]
        self.contour[:, 0, 0] += x_start
        self.contour[:, 0, 1] += y_start

    def draw(self):
        """draw for visualization
        To be implemented later
        """
        pass

    def insert(self, contour, bbox):
        """Insert a new region into the tree
        contour: opencv contour of the region
        bbox: inner library's representation of the bbox
        NOTE: NOT USABLE YET, only usable in cases where the input is correct
        which is not the output from opencv currently
        """
        # modified binary tree insert operation
        area = cv2.contourArea(contour)
        if area > self.area:
            raise ValueError("MSER::TreeMSER::Inserting a node larger than\
                             its parent")
        list_contain = []
        list_contained = []
        for i, child in enumerate(self.children):
            """
            contained, swapped = bbox_operations.check_bbox_contains_each_other(
                bbox, child.bbox)
            """
            contain = cv2.pointPolygonTest(contour, tuple(child.contour[0]),
                                           False)
            contained = cv2.pointPolygonTest(child.contour, tuple(contour[0]),
                                             False)
            if contain >= 0 and contained < 0 and area > child.area:
                list_contain.append(i)
            elif contained >= 0 and contain < 0 and area < child.area:
                list_contained.append(i)
            """
            if contained:
                if area > child.area and not swapped:
                    list_contain.append(i)
                elif area < child.area and swapped:
                    list_contained.append(i)
            """
        if len(list_contain) > 0:
            if len(list_contained) > 0:
                print("Child bbox: ", self.children[list_contain[0]].bbox,
                      self.children[list_contain[0]].area)
                print("This: ", bbox, " ", area)
                print("Parent bbox: ", self.children[list_contained[0]].bbox,
                      self.children[list_contained[0]].area)
                raise ValueError("MSER::TreeMSER::Tree Property Not Preserved")
            new_children_self = [
                self.children[i] for i in range(len(self.children))
                if i not in list_contain
            ]
            new_children_self.append(TreeMSER(self, area, contour, bbox))
            children_new_node = [self.children[i] for i in list_contain]
            for i in range(len(children_new_node)):
                children_new_node[i].parent = new_children_self[-1]
            new_children_self[-1].children = children_new_node
            self.children = new_children_self
        else:
            if len(list_contained) > 0:
                if len(list_contained) > 1:
                    print("This: ", bbox, " ", area)
                    print("Parent bbox: ",
                          self.children[list_contained[0]].bbox,
                          self.children[list_contained[0]].area)

                    raise ValueError(
                        "MSER::TreeMSER::Tree Property Not Preserved"
                    )
                self.children[list_contained[0]].insert(contour, bbox)
            else:
                self.children.append(TreeMSER(self, area, contour, bbox))


# Calculate val of a node
def calculate_var(node, theta1=0.03, theta2=0.08, alpha_min=0.7,
                  alpha_max=1.2):
    # Presuming MSERed & after filtered, delta = 1
    if node.parent:
        val = node.var
        aspect_ratio = node.aspect_ratio
        if aspect_ratio > alpha_max:
            val -= theta1*(aspect_ratio - alpha_max)
        elif aspect_ratio < alpha_min:
            val -= theta2*(alpha_min - aspect_ratio)
        return val
    else:
        return 1


def link_children(node1, node2):
    for i in range(len(node2.children)):
        node2.children[i].parent = node1
    node1.children.extend(node2.children)
    node2.children = []


def linear_reduction(T):
    if len(T.children) == 0:
        return T
    elif len(T.children) == 1:
        c = linear_reduction(T.children[0])
        if calculate_var(T) <= calculate_var(c):
            # link children
            link_children(T, c)
            return T
        else:
            return c
    else:
        for i in range(len(T.children)):
            linear_reduct = linear_reduction(T.children[i])
            link_children(T, linear_reduct)
        return T


def tree_accumulation(T):
    if len(T.children) > 2:
        C = []
        for c in T.children:
            C.extend(tree_accumulation(c))
        vals = [calculate_var(c) for c in C]
        if calculate_var(T) <= min(vals):
            T.children = []
            return [T]
        else:
            return C
    else:
        return [T]


def mser_to_tree(imggray):
    """
    Input:
        gray scale image
    Output:
        Tree containing all mser regions big to small
    """
    # delta, min_area, max_area, max_variation, min_diversity, max_evolution
    # area_threshold, min_margin, edge_blur_size
    mser = cv2.MSER_create(5, 20, 14400, 0.25, 0.2, 200, 1.01, 0.003, 5)
    msers, cv_bboxs = mser.detectRegions(imggray)
    # cv_bboxs: x, y, w, h
    # Start building tree hierarchical
    root = TreeMSER(None)
    root.area = np.prod(imggray.shape)
    for i in range(len(msers)):
        root.insert(msers[i], bbops.get_bbox_cv(
            cv_bboxs[i][0], cv_bboxs[i][1], cv_bboxs[i][2], cv_bboxs[i][3]))
    return root


def mser_prunning(root):
    root = linear_reduction(root)
    C = tree_accumulation(root)
    return C


def draw_contour_MSER(imggray, C):
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
              (0, 255, 255), (255, 0, 255)]
    img = cv2.cvtColor(imggray, cv2.COLOR_GRAY2BGR)
    for i, c in enumerate(C):
        if c.contour is None:
            continue
        print(c.contour)
        cv2.drawContours(img, [c.contour], -1, colors[i % len(colors)], 3)
    return img
