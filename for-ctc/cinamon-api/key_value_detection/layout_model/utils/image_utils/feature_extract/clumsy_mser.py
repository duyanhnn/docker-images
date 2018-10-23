from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
import numpy as np
from matplotlib import pyplot as plt
from ..local_imutils import (
    connected_component_analysis_sk_with_labels as cca_sk)
from ..bboxs_tools import bbox_operations as bbops
from .mser import TreeMSER


def find_sub_connected_component(imggray, i,
                                 offset_x=0, offset_y=0,
                                 old_area=999999999, step=10, debug=False):
    """Recursive function to find mser hierarchical component tree
    """
    imggray_filtered = imggray > i
    if debug:
        plt.subplot(1, 2, 1)
        plt.imshow(imggray)
        plt.subplot(1, 2, 2)
        plt.imshow(imggray_filtered)
        plt.show()

    labels, bboxs = cca_sk(imggray_filtered)
    unique_labels = np.array(sorted(np.unique(labels)))
    unique_labels = unique_labels[unique_labels > 0]
    if len(unique_labels) == 1:
        i += step
        if i < 255:
            return find_sub_connected_component(imggray, i,
                                                offset_x, offset_y,
                                                old_area,
                                                step=step,
                                                debug=debug)

    nodes = []
    if i < 255:
        i += step
    for j, l in enumerate(unique_labels):
        new_node = TreeMSER()
        labels_node = np.zeros_like(labels)
        labels_node_mask = (labels == l)
        labels_node[labels_node_mask] = imggray[labels_node_mask]
        if debug:
            plt.subplot(1, 2, 1)
            plt.imshow(labels)
            plt.title('labels')
            plt.subplot(1, 2, 2)
            plt.imshow(labels_node)
            plt.title('labels node '+str(l))
            plt.show()
        new_node.area = np.sum(labels_node_mask)
        # Border following
        labels_node_pixels = np.where(labels_node_mask)
        # Changing to x, y
        labels_node_pixels = np.transpose(
            np.array([labels_node_pixels[1], labels_node_pixels[0]]))

        new_node.sample_pixel = labels_node_pixels[0][:]
        new_node.sample_pixel[0] += offset_x
        new_node.sample_pixel[1] += offset_y
        labels_node_pixels = np.transpose(labels_node_pixels)
        smaller_image = bbops.crop_bbox(labels_node, bboxs[j])
        """
        new_node.contour = border_following_clockwise(labels_node_pixels)
        new_node.contour[:, 0] += offset_x
        new_node.contour[:, 1] += offset_y
        """
        new_node.bbox = bboxs[j][:]
        new_node.bbox[:, 0] += offset_x
        new_node.bbox[:, 1] += offset_y
        new_node.level = i - step
        nodes.append(new_node)
        # create a new node then call this function recursively
        bboxs[j][:, 0] += offset_x
        bboxs[j][:, 1] += offset_y
        x_start, _, y_start, _ = bbops.get_x_y_start_stop(bboxs[j])
        if debug:
            plt.imshow(smaller_image)
            plt.title('bbox'+str(l))
            plt.show()
        if i <= 255:
            children = find_sub_connected_component(smaller_image, i,
                                                    x_start, y_start,
                                                    nodes[j].area,
                                                    step=step,
                                                    debug=debug)
            if debug and len(children) > 0:
                print("Children Length: ", len(children))
            for k in range(len(children)):
                children[k].parent = nodes[j]
            nodes[j].children = children
    return nodes


def mser_tree(imggray, debug=False):
    """
    """
    img_area = np.prod(imggray.shape)
    root = TreeMSER(None, img_area)
    nodes = find_sub_connected_component(imggray, 0, old_area=root.area,
                                         debug=debug)
    for i in range(len(nodes)):
        nodes[i].parent = root
    root.children = nodes
    return root


def calculate_var_tree(node, delta=2):
    node.val = 9999
    to_modify = True
    parent = node
    for i in range(delta):
        if node.parent:
            parent = node.parent
        else:
            to_modify = False
            break
    if to_modify:
        node.val = abs(parent.area - node.area)/node.area
    for i in range(len(node.children)):
        calculate_var_tree(node.children[i])
