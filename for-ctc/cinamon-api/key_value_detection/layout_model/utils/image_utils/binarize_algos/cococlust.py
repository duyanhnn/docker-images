from __future__ import division
import numpy as np
import cv2

from scipy.spatial.distance import cdist
from scipy.cluster.vq import kmeans2
from numpy import linalg as LA
from ...generic_utils import sorted_index
from .common_sg import (filter_out_big_and_small, get_merged_edges,
                        filter_out_childs,
                        get_median_background)
from ..bboxs_tools.bbox_operations import (get_area_bounding_box,
                                           get_bbox_x_y_w_h)
from ..gradients_tools.edges_utils import (get_blurred_contour,
                                           get_normals_contour_smooth,
                                           get_normals_contour)


def get_sample_pixels(image_array, locations, norms, num_sample_plus=3,
                      num_sample_minus=3, debug=False):
    """
    Input:
        Image array: gray or not
        Location: Location of contours
        norms: norms of contours
        num_sample_plus: into outer
        num_sample_minus: into inner
    Output:
        colors: Nx3 if colored or Nx1 if gray
    TESTED!!!! (Contour: x, y)
    """
    indices = []
    if debug:
        print("input to sample pixel shape: ", image_array.shape)
        print("contours: ", locations)
        print("norms: ", norms)
    for i in range(num_sample_plus):
        indices.extend(locations+norms*(i+1))
    for i in range(num_sample_minus):
        indices.extend(locations+norms*(-i-1))
    indices = np.array(indices).astype('int')
    print("indices: ", indices)
    indices[:, 0] = np.minimum(
        indices[:, 0],
        np.ones(indices.shape[0])*(image_array.shape[1]-1)
    )
    indices[:, 1] = np.minimum(
        indices[:, 1],
        np.ones(indices.shape[0])*(image_array.shape[0]-1)
    )
    indices[:, 0] = np.maximum(
        indices[:, 0],
        np.zeros(indices.shape[0])*(image_array.shape[1]-1)
    )
    indices[:, 1] = np.maximum(
        indices[:, 1],
        np.zeros(indices.shape[0])*(image_array.shape[0]-1)
    )

    unique_colors = np.unique(image_array[indices[:, 1], indices[:, 0]], axis=0)
    if debug:
        print("First Sample colors shape: ", unique_colors[0].shape)
        print("First Sample colors: ", unique_colors[0])
    return list(unique_colors)


def transform_color_space(color_samples, white_ref=255):
    """
    Color samples: Nx3 matrix, RGB
    """
    transform_xyz_mat = np.array([[0.490, 0.177, 0],
                                  [0.310, 0.812, 0.01],
                                  [0.2, 0.011, 0.990]])
    white_ref_rgb = np.array([[white_ref, white_ref, white_ref]])
    white_ref = np.matmul(white_ref_rgb, transform_xyz_mat)[0]

    color_samples = np.matmul(color_samples, transform_xyz_mat)
    for i in range(3):
        color_samples[:, i] /= white_ref[i]

    mask = color_samples > 0.008856
    invmask = np.invert(mask)

    color_samples[mask] = np.cbrt(color_samples[mask])
    color_samples[invmask] = 7.787*color_samples[invmask] + 0.138

    new_color_samples = np.zeros(color_samples.shape)
    new_color_samples[:, 0] = 116*color_samples[:, 1] - 16
    new_color_samples[:, 1] = 500*(color_samples[:, 0] - color_samples[:, 1])
    new_color_samples[:, 2] = 200*(color_samples[:, 1] - color_samples[:, 2])

    return new_color_samples


def reverse_transform_color_space(lab_color_samples, white_ref=255):
    """
    CIELab back to rgb space
    """
    transform_xyz_mat = np.array([[0.490, 0.177, 0],
                                  [0.310, 0.812, 0.01],
                                  [0.2, 0.011, 0.990]])
    white_ref_rgb = np.array([[white_ref, white_ref, white_ref]])
    white_ref = np.matmul(white_ref_rgb, transform_xyz_mat)[0]
    xyz_color_samples = np.zeros(lab_color_samples.shape)
    xyz_color_samples[:, 1] = (lab_color_samples[:, 0]+16)/116
    xyz_color_samples[:, 0] = xyz_color_samples[:, 1] +\
        lab_color_samples[:, 1]/500
    xyz_color_samples[:, 2] = xyz_color_samples[:, 1] -\
        lab_color_samples[:, 2]/200
    mask = xyz_color_samples > 0.206897
    invmask = np.invert(mask)
    xyz_color_samples[mask] = xyz_color_samples[mask] ** 3
    xyz_color_samples[invmask] = 0.128419*(xyz_color_samples[invmask] -
                                           0.137931)

    xyz_color_samples[:, 0] = xyz_color_samples[:, 0]*white_ref[0]
    xyz_color_samples[:, 1] = xyz_color_samples[:, 1]*white_ref[1]
    xyz_color_samples[:, 2] = xyz_color_samples[:, 2]*white_ref[2]

    rgb_color_samples = np.matmul(xyz_color_samples, LA.inv(transform_xyz_mat))
    return rgb_color_samples


def get_sample_colors(colored_image_array, edges, window_size=5, skip=6,
                      num_samples_each_side=3, debug=False):
    """
    This should be optimized in the future
    Input:
        colored_image_array: original image
        edges: canny filtered
        window_size: mean size
        skip: skip size
        num_samples_each_side: at each point, takes this variables and samples
        color each size
    Output:
        Nx3 Color from CIELab space
    """
    im2, contours, hierarchy = cv2.findContours(edges,
                                                cv2.RETR_TREE,
                                                cv2.CHAIN_APPROX_NONE)
    if window_size % 2 == 0:
        window_size += 1
    mean_signal = np.ones(window_size)/window_size
    color_samples = []
    for contour in contours:
        contour = contour.reshape((contour.shape[0], 2))
        # Generate smoothed contour
        mean_contour = get_blurred_contour(contour, window_size, mean_signal)
        # Generate normal vectors
        mean_contour, norms = get_normals_contour_smooth(mean_contour, skip)
        # Takes outer and inner color sample
        color_samples.extend(get_sample_pixels(colored_image_array,
                                               mean_contour, norms,
                                               num_samples_each_side,
                                               num_samples_each_side,
                                               debug))
    color_samples = np.array(color_samples).astype('float')
    color_samples = np.unique(color_samples, axis=0)
    if debug:
        print("Color samples total: ", len(color_samples))
    # Swap blue to red, BGR -> RGB
    color_samples[:, [0, 2]] = color_samples[:, [2, 0]]
    return transform_color_space(color_samples)


def kasar_lab_cluster(colors_lab, threshold=45, debug=False):
    """
    Cluster according to the cococlust paper (kasar 2011)
    Input:
        colors_lab: colors in CIELAB space
        threshold: Ts in paper, the larger, the less centroids
    Output:
        Centroids.
    """
    clusters = [colors_lab[0]]
    count = 1
    for color_test in colors_lab[1:]:
        print("num_clusters: ", count)
        for j in range(count):
            if LA.norm(clusters[j] - color_test) <= threshold:
                clusters[j] = (color_test+clusters[j])/2
                break
            else:
                count += 1
                clusters.append(color_test)
    clusters = np.array(clusters)
    if debug:
        print("Number of clusters: ", clusters.shape[0])
    return clusters


def kasar_lab_cluster_modified(colors_lab, threshold=45, debug=False):
    """
    Cluster according to the cococlust paper (kasar 2011)
    Modified to avoid exploded adding cluster problem
    Input:
        colors_lab: colors in CIELAB space
        threshold: Ts in paper, the larger, the less centroids
    Output:
        Centroids.
    """
    clusters = [colors_lab[0]]
    count = 1
    for color_test in colors_lab[1:]:
        print("num_clusters: ", count)
        to_be_add = True
        for j in range(count):
            if LA.norm(clusters[j] - color_test) <= threshold:
                clusters[j] = (color_test+clusters[j])/2
                to_be_add = False
                break
        if to_be_add:
            count += 1
            clusters.append(color_test)
    clusters = np.array(clusters)
    if debug:
        print("Number of clusters: ", clusters.shape[0])
    return clusters


def rgb_kasar_lab_cluster_modified(colors_lab, threshold=60, debug=False):
    """
    Cluster according to the cococlust paper (kasar 2011)
    Modified to avoid exploded adding cluster problem
    Input:
        colors_lab: colors in RGB space
        threshold: Ts in paper, the larger, the less centroids
    Output:
        Centroids.
    """
    clusters = [colors_lab[0]]
    count = 1
    for color_test in colors_lab[1:]:
        print("num_clusters: ", count)
        to_be_add = True
        for j in range(count):
            if LA.norm(clusters[j] - color_test) <= threshold:
                clusters[j] = (color_test+clusters[j])/2
                to_be_add = False
                break
        if to_be_add:
            count += 1
            clusters.append(color_test)
    clusters = np.array(clusters)
    if debug:
        print("Number of clusters: ", clusters.shape[0])
    return clusters


def assigns_to_clusters(colors, clusters, debug=False):
    distances = []
    if debug:
        print("Assigns to cluster colors shape: ", colors.shape)
        print("Assigns to cluster clusters shape: ", clusters.shape)
    for cluster in clusters:
        distances.append(cdist(colors, np.array([cluster])))
    indices = np.argmin(distances, axis=0)
    return indices, distances


def calculate_clusters_final(colors, clusters, threshold=45, debug=False):
    """
    Performing assignment to clusters
    """
    divide_threshold = threshold*0.75
    indices, distances = assigns_to_clusters(colors, clusters, debug)
    new_clusters = []
    for i, cluster in enumerate(clusters):
        mask = (indices == i)
        mask = mask.reshape(mask.shape[0])
        distances_this_clusters = distances[i][mask]
        if debug:
            print("Colors: ", colors.shape)
            print("Mask :", mask.shape)
            print("Total distances: ", distances[i].shape)
            print("Distance current clusters: ", distances_this_clusters.shape)
        colors_this_clusters = colors[mask]
        if len(distances_this_clusters) == 0:
            continue
        start_index = np.argmax(distances_this_clusters)
        if distances_this_clusters[start_index] > divide_threshold:
            start_color = colors_this_clusters[start_index]
            matrix_starts = np.array([start_color, cluster])
            # Perform kmeans, 2 centroids
            centroids, _ = kmeans2(colors[mask], matrix_starts,
                                   minit='matrix')
            new_clusters.extend(centroids)
        else:
            new_clusters.append(cluster)
    new_clusters = np.array(new_clusters)
    new_clusters, _ = kmeans2(colors, new_clusters, minit='matrix')
    if debug:
        print("Num clusters: ", new_clusters.shape[0])
    return new_clusters


def cococlust(colored_image_array, thres1=50, thres2=200, Ts=45, debug=False):
    colored_image_array = cv2.medianBlur(colored_image_array, 5)
    edges = get_merged_edges(colored_image_array, thres1, thres2, debug)
    colored_img_shape = colored_image_array.shape
    img_shape = edges.shape
    edges = filter_out_big_and_small(edges)

    # Sample colors should not be based on all labels
    lab_colors_samples = get_sample_colors(
        colored_image_array, edges, window_size=5,
        skip=6, num_samples_each_side=3, debug=debug)
    flat_colored = colored_image_array.reshape((np.prod(img_shape), 3))
    # Cluster color samples
    clusters = kasar_lab_cluster(lab_colors_samples)
    clusters = calculate_clusters_final(lab_colors_samples, clusters, Ts, debug)

    # Assign clusters
    flat_colored[:, [0, 2]] = flat_colored[:, [2, 0]]

    flat_colored_xyz = transform_color_space(flat_colored, white_ref=255)
    indices, _ = assigns_to_clusters(flat_colored_xyz, clusters)
    rgb_clusters = reverse_transform_color_space(clusters, white_ref=255)
    flat_colored = rgb_clusters[indices]
    flat_colored = flat_colored.reshape((flat_colored.shape[0], 3))
    if debug:
        print("flat colored shape: ", flat_colored.shape)
    flat_colored[:, [0, 2]] = flat_colored[:, [2, 0]]
    colored_img_clustered = flat_colored.reshape(colored_img_shape)
    if debug:
        print("Completed clustering")
        cv2.imwrite("debug_clustered.jpg", colored_img_clustered)
    gray = cv2.cvtColor(colored_img_clustered.astype('uint8'),
                        cv2.COLOR_BGR2GRAY)

    # Find edges and contours, sencond times
    edges = get_merged_edges(colored_img_clustered.astype('uint8'),
                             thres1, thres2, debug)

    edges = filter_out_big_and_small(edges)
    im2, contours, hier = cv2.findContours(edges,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_NONE)
    # Estimate contours bboxs
    contours = [cnt.reshape((cnt.shape[0], 2)) for cnt in contours]
    bboxs = [get_bbox_x_y_w_h(*cv2.boundingRect(cnt))
             for cnt in contours]
    areas = [get_area_bounding_box(bbox) for bbox in bboxs]
    sorted_index_text_only = sorted_index(areas, inverse=True)
    # Filter
    sorted_index_text_only_final = filter_out_childs(bboxs,
                                                     sorted_index_text_only,
                                                     debug)

    # Estimate foreground and background for each connected component
    binary_img = np.ones(img_shape)*255
    for index in sorted_index_text_only_final:
        contour = contours[index].astype('int')
        i_foreground = np.mean(gray[contour[:, 1], contour[:, 0]])
        mean_contour, normals = get_normals_contour(contour)
        pixels = get_sample_pixels(gray, mean_contour, normals, 4, 0, debug)
        if len(pixels) == 0:
            i_background = get_median_background(gray, bboxs[index])
        else:
            i_background = np.median(pixels)

        # Only for pixels inside contour
        cimg = np.zeros(img_shape)
        contour = contour.reshape([contour.shape[0], 1, 2])
        cv2.drawContours(cimg, [contour], 0, color=255, thickness=-1)
        pts = np.where(cimg == 255)
        if debug:
            print("Foreground: ", i_foreground, ",Background: ", i_background)
        if i_foreground > i_background:
            binary_img[pts[0], pts[1]] = (
                (gray[pts[0], pts[1]] < i_foreground) *
                (binary_img[pts[0], pts[1]] != 0))*255
        else:
            binary_img[pts[0], pts[1]] = (
                (gray[pts[0], pts[1]] > i_foreground) *
                (binary_img[pts[0], pts[1]] != 0))*255
    if debug:
        print("Done binarizing")
        cv2.imwrite("debug_binary_cococlust_" + str(thres1) +
                    "_" + str(thres2) + ".jpg", binary_img)
    return binary_img
