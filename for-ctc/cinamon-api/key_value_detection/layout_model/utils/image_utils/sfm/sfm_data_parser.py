from __future__ import unicode_literals
from __future__ import print_function
import json
import os
import cv2
import numpy as np
from numpy.linalg import inv as inverse
from numpy import array, transpose, matmul, zeros, identity, sqrt
from .multiview_rectification import get_A_b, update_W


def customized_search_list(the_list, key):
    for i, list_elem in enumerate(the_list):
        if list_elem['key'] == key:
            break
    return i, list_elem


def load_sfm_data_mvg(filepath, debug=True):
    """
    Load sfm data pass, transform points according to views with most points
    return those points from the perspective of view points
    """
    sfm_data = None
    with open(filepath, 'r') as sfm_data_file:
        sfm_data = json.load(sfm_data_file)
    try:
        image_dir = sfm_data['root_path']
        cloud_points = sfm_data['structure']
        views_transform = sfm_data['extrinsics']
        views_data = sfm_data['views']

    except ValueError:
        print("Wrong sfm_data json format\n")
        return []

    views_points = {}
    feats = {}
    for view_info in views_data:
        views_points[int(view_info['key'])] = []
        feats[int(view_info['key'])] = []
    print(views_points)
    for point in cloud_points:
        views_correspond = point['value']['observations']
        for view in views_correspond:
            views_points[int(view['key'])].append(point['value']['X'])
            feats[int(view['key'])].append(view['value']['x'])
    # Find views that has maximum number of points
    max_view_key = max(views_points,
                       key=(lambda key: len(views_points[key])))
    print("max view key: ", max_view_key)
    view_points = np.ones((4, array(views_points[max_view_key]).shape[0]))
    view_points[0:3, :] = transpose(array(views_points[max_view_key]))
    print("view points shape: ", view_points.shape)
    if debug:
        if view_points.shape[0] < 600:
            print("Warning: Too few points for max view")

    i, view_transform = customized_search_list(views_transform, max_view_key)
    if i == len(views_transform):
        raise ValueError('Bad sfm_data file: missing views transform\n')
        return []

    i, view_image = customized_search_list(sfm_data['views'], max_view_key)
    image_path = os.path.join(
        image_dir,
        view_image['value']['ptr_wrapper']['data']['filename'])
    i, cam_transform = customized_search_list(
        sfm_data['intrinsics'],
        view_image['value']['ptr_wrapper']['data']['id_intrinsic'])
    if i == len(views_data):
        raise ValueError('Bad sfm_data file: missing views data\n')
        return []

    # To camera view
    rotation = array(view_transform['value']['rotation']).transpose()
    translation = array(view_transform['value']['center']).reshape((3))
    camera_view_transform = np.zeros((3, 4))
    camera_view_transform[:3, :3] = rotation
    camera_view_transform[:3, 3] = translation
    camera_view_points = matmul(camera_view_transform, view_points)
    if debug:
        print("Camera view points: ", camera_view_points)
    feats[max_view_key] = np.array(feats[max_view_key])
    return [image_path, cam_transform, view_transform, camera_view_points,
            feats[max_view_key]]


def permutation_matrix_from_point(width, height, points, is_column=False):
    """
    Construct permutation matrix from points
    input:
        width: the width of the image
        height: the height of the image
        points: the N observed points
        is_column: True if the points are column vertices
    output:
        permutation matrix: [N,(width*height)] permutation matrix
    """
    permutation_list = []
    if len(points) == 0:
        raise ValueError("Permutation Matrix From points:\
                         points set must not be empty\n")
    for point in points:
        # take indices in meshgrid
        print(point)
        permutation_list.append(int(width*point[1]) + int(point[0]))
    permutation_matrix = zeros((len(points), width*height))
    for i in range(permutation_matrix.shape[0]):
        permutation_matrix[i, permutation_list[i]] = 1
    return permutation_matrix


def sparse_derivative_matrix_image(width, height):
    I = width*height
    D = zeros((2*I, I))
    for i in range(I):
        for j in range(I):
            if i == j:
                D[i, j] = 2
                D[i+I, j] = 2
                continue
            y_i = int(i/width)
            x_i = i - y_i*width
            y_j = int(j/width)
            x_j = j - y_j*width
            if x_i == x_j or y_i == y_j:          # Above, below, left, right
                D[i, j] = -1
                D[i+I, j] = -1
                continue
    return D


def ridge_aware_optimization(image, image_points, feats,
                             lamb=10**-5, eps=10**-8,
                             alpha=10**-8):
    """
    Ridge aware optimization, based on Multiview Rectification of Folded
    Documents
    input:
        image: input image
        image_points: observed points in image view
    output:
        z_map
    """
    # -1. reshape for smaller image
    scale = max(image.shape[0], image.shape[1])
    scale = scale/100.0
    image = cv2.resize(image,
                       (int(image.shape[1]/scale), int(image.shape[0]/scale)))
    # 0. Construct meshgrid indices
    height = image.shape[0]
    width = image.shape[1]
    I = width*height
    image_points = image_points.transpose()
    feats /= scale
    N = len(image_points)
    observed_z = image_points[:, 2].reshape((N, 1))
    # 1. Construct permutation matrix
    permutation_mat = permutation_matrix_from_point(width, height, feats)
    # 2. Construct D matrix
    D = sparse_derivative_matrix_image(width, height)
    # 3. Construct Weight matrix
    W = identity(N)
    # 4. Optimize
    # step 0, calculate A & B
    lower_A = sqrt(lamb)*D
    A, b = get_A_b(N, I, W, permutation_mat, lower_A, observed_z)
    # step 1 update z
    z_image = matmul(matmul(inverse(matmul(A.transpose(), A) + alpha),
                            A.transpose()), b)
    W = update_W(W, permutation_mat, z_image, observed_z, eps)
    z_new = matmul(matmul(inverse(matmul(A.transpose(), A) + alpha),
                   A.transpose()), b)
    distance = np.linalg.norm(z_new-z_image, ord=2)
    while(distance > eps):
        z_image = z_new
        A, b = get_A_b(N, I, W, permutation_mat, lower_A, observed_z)
        W = update_W(W, permutation_mat, z_image, observed_z, eps)
        z_new = matmul(matmul(inverse(matmul(A.transpose(), A) + alpha),
                       A.transpose()), b)
        distance = np.linalg.norm(z_new-z_image, ord=2)

    # we may have to get another times, but right now just export it first
    # 5. Export to another json

    return z_new.reshape(image.shape[:2])
