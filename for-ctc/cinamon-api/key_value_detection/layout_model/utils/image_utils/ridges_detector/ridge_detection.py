import cv2
import numpy as np
from ..gradients_tools.difftools import hessian_from_grad, hessian
from ..local_imutils import invert_color_image
from ...generic_utils import frange
from ..filter_tools.filters import get_line_averaging_filter
from . import ani
from scipy import ndimage


def cal_R_dx_dy(x, y, dx, dy, lambda_s, lambda_l, gx, gy, C):
    r = 0
    if lambda_s[y, x] < 0 and abs(lambda_s[y, x]) > abs(lambda_l[y, x]):
        if lambda_s[
            y+dy,
            x+dx
        ] < 0 and abs(lambda_s[y+dy, x+dx]) > abs(lambda_l[y+dy, x+dx]):
            grad_x_y = np.array([gx[y, x], gy[y, x]])
            grad_dx_dy = np.array([gx[y+dy, x+dx], gy[y+dy, x+dx]])
            if np.dot(grad_x_y, grad_dx_dy) < np.dot(C[y, x, :],
                                                     C[y+dy, x+dx, :]):
                if np.dot(grad_x_y, C[y, x, :]
                          )*np.dot(
                              grad_dx_dy,
                              C[y+dy, x+dx, :])*np.dot(C[y, x, :],
                                                       C[y + dy, x+dx]) < 0:
                    r = 1
    return r


def cal_R(x, y, lambda_s, lambda_l, gx, gy, gdc):
    dx_ys = []
    if x > 0:
        dx_ys.append((-1, 0))
    if y > 0:
        dx_ys.append((0, -1))
    if x < lambda_s.shape[1]-1:
        dx_ys.append((1, 0))
    if y < lambda_s.shape[0]-1:
        dx_ys.append((0, 1))
    r = 0
    for dx_y in dx_ys:
        r = cal_R_dx_dy(x, y, dx_y[0], dx_y[1],
                        lambda_s, lambda_l, gx, gy, gdc)
        if r > 0:
            break
    return r


def detect_ridge_from_neighbors(lambda_s, lambda_l, gx, gy, gdc):
    ridges = np.zeros(lambda_s.shape)
    for y in range(lambda_s.shape[0]):
        for x in range(lambda_s.shape[1]):
            ridges[y, x] = cal_R(x, y, lambda_s, lambda_l, gx, gy, gdc)
    return ridges


def cal_image_grad_gdc(image_smoothed):
    """Steps:
        1. Calculate hessian matrix and gradient
        2. Calculate greatest downward curvature
        3. Check condition to mark pixel as ridge or not
    """
    image_smoothed = image_smoothed.astype('int8')
    gy, gx = np.gradient(image_smoothed)
    hessian_img = hessian_from_grad(gx, gy)
    h, w = hessian_img.shape[0], hessian_img.shape[1]
    print(h, w)
    print(gx.shape)
    gdc_image = np.zeros((h, w, 2))
    lambda_s = np.zeros((h, w))
    lambda_l = np.zeros((h, w))
    for y in range(h):
        for x in range(w):
            eig_vals, eig_vecs = np.linalg.eig(hessian_img[y, x])
            min_indice = np.argmin(eig_vals)
            gdc_image[y, x, :] = eig_vecs[min_indice]/np.linalg.norm(
                eig_vecs[min_indice])
            lambda_s[y, x] = eig_vals[min_indice]
            lambda_l[y, x] = eig_vals[1-min_indice]
    return gx, gy, gdc_image, lambda_s, lambda_l


def detect_ridge_grad(image_smoothed):
    """Steps:
        1. Calculate hessian matrix and gradient
        2. Calculate greatest downward curvature
        3. Check condition to mark pixel as ridge or not
    """
    gx, gy = np.gradient(image_smoothed)
    hessian_img = hessian_from_grad(gx, gy)
    h, w = hessian_img.shape[0], hessian_img.shape[1]
    print(h, w)
    print(gx.shape)
    ridge_candidates = []
    for y in range(h):
        for x in range(w):
            eig_vals, eig_vecs = np.linalg.eig(hessian_img[y, x])
            min_indice = np.argmin(eig_vals)
            norm_eig_vec = eig_vecs[min_indice]/np.linalg.norm(
                eig_vecs[min_indice])
            if eig_vals[min_indice] < 0:
                norm_grad_vec = np.array([gx[y, x],
                                          gy[y, x]])/np.linalg.norm(
                                              np.array(
                                                  [gx[y, x],
                                                   gy[y, x]]))

                if np.abs(np.linalg.norm(
                    np.dot(norm_eig_vec, norm_grad_vec)
                 )) == 0:
                    ridge_candidates.append((x, y))
    return ridge_candidates


def detect_ridge(image_smoothed):
    """Steps:
        1. Calculate hessian matrix and gradient
        2. Calculate greatest downward curvature
        3. Check condition to mark pixel as ridge or not
    """
    gx, gy = np.gradient(image_smoothed)
    hessian_img = hessian(image_smoothed)
    hessian_img = np.transpose(hessian_img, (2, 3, 0, 1))
    h, w = hessian_img.shape[0], hessian_img.shape[1]
    print(h, w)
    print(gx.shape)
    ridge_candidates = []
    for y in range(h):
        for x in range(w):
            eig_vals, eig_vecs = np.linalg.eig(hessian_img[y, x])
            min_indice = np.argmin(eig_vals)
            norm_eig_vec = eig_vecs[min_indice]/np.linalg.norm(
                eig_vecs[min_indice])
            if eig_vals[min_indice] < 0:
                if np.linalg.norm(
                    np.dot(norm_eig_vec, np.array(
                        [gy[y, x], gx[y, x]]))
                 ) < 0.00001:
                    ridge_candidates.append((x, y))
    return ridge_candidates


def textline_structure_enhancement(imggray, l_start, l_stop, l_step,
                                   theta_start=-np.pi/4,
                                   theta_stop=np.pi/4, theta_step=0.2):
    """Textline structure enchancement like in the paper "Generic Method
    For Document Layout Analysis",
    Step1: Gaussian blur
    Step2: Line Averaging filtering
    Step3: Gaussian blur again
    Future improvement needed: Gaussian blur based on image properties:
        example: CCA Size
    """
    imggray = invert_color_image(imggray)
    blur = ndimage.gaussian_filter(imggray, 8)
    filters = []
    for l in range(l_start, l_stop, l_step):
        for theta in frange(theta_start, theta_stop, theta_step):
            filters.append(get_line_averaging_filter(l, theta))
    filtereds = []
    for line_avg_filter in filters:
        filtereds.append(cv2.filter2D(blur, -1, line_avg_filter))

    filtereds = np.array(filtereds)
    filtered_shapes = filtereds.shape[1:]
    num_filtereds = filtereds.shape[0]

    filtereds = np.reshape(filtereds, (num_filtereds, np.prod(filtered_shapes)))
    print(filtereds.shape)
    filtereds = np.amax(filtereds, axis=0)
    filtereds = ndimage.gaussian_filter(np.reshape(filtereds, filtered_shapes),
                                        8)
    print(np.amax(filtereds))
    filtereds = filtereds*255.0/np.amax(filtereds)
    return filtereds


def textline_structure_enhancement_anigauss(
        imggray, sigx_start, sigx_stop, sigx_step,
        sigy_start, sigy_stop, sigy_step,
        theta_start=-45,
        theta_stop=45, theta_step=2):
    """Textline structure enchancement like in the paper "Generic Method
    For Document Layout Analysis",
    Step1: Anisotropic Gaussian Blur
    Step2: Gaussian blur again
    Future improvement needed: Gaussian blur based on image properties:
        example: CCA Size
    """
    imggray = invert_color_image(imggray)
    filtereds = []
    for sigx in frange(sigx_start, sigx_stop, sigx_step):
        for sigy in frange(sigy_start, sigy_stop, sigy_step):
            for theta in frange(theta_start, theta_stop, theta_step):
                directional_blur = ani.anigauss(imggray, sigy, sigx, theta)
                filtered_shapes = directional_blur.shape
                directional_blur.reshape(np.prod(directional_blur.shape))
                if len(filtereds) == 0:
                    filtereds = directional_blur
                else:
                    filtereds = np.maximum(directional_blur, filtereds)

    filtereds = np.reshape(filtereds, filtered_shapes)
    filtereds = ndimage.gaussian_filter(np.reshape(filtereds, filtered_shapes),
                                        8)
    filtereds = filtereds*255.0/np.amax(filtereds)
    return filtereds
