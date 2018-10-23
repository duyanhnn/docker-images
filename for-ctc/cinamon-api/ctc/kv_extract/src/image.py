import cv2
import os
import sys
import numpy as np

from PIL import Image
from PIL.ExifTags import TAGS

SCAN_IMAGE_TYPE = 1
PHOTO_IMAGE_TYPE = 2

PALETTES_FIXED = [(224, 243, 250), (252, 246, 222), (70, 183, 139), (231, 231, 231), (153, 153, 153), (144, 182, 37),
                  (255, 246, 241), (247, 128, 96), (68, 179, 190), (84,
                                                                    161, 217), (255, 253, 232), (137, 124, 204),
                  (238, 238, 238), (243, 161, 0), (254, 216, 143), (255,
                                                                    235, 198), (255, 254, 203), (145, 180, 54),
                  (155, 195, 81), (255, 254, 203), (55, 121,
                                                    207), (255, 205, 206), (113, 162, 221),
                  (218, 240, 251), (244, 216, 2), (64, 101, 179), (243, 240, 233), (244, 244, 244), (0, 51, 153)]

THRESHOLD_TO_INVERT = 180
THRESHOLD_TO_FILL = 10
THRESHOLD_TO_FLOODFILL = 20
THRESHOLD_TO_DRAW_REG = 25
PERCENT_ACCEPTION = 1


def variance_of_laplacian(image):
    # compute the Laplacian of the image and then return the focus
    # measure, which is simply the variance of the Laplacian
    return cv2.Laplacian(image, cv2.CV_64F).var()


def isblur(img_dir, threse=10):
    image = cv2.imread(img_dir)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = variance_of_laplacian(gray)
    if fm < threse:
        return True
    return False


def detect_image_type(img_file):
    image_info = {}
    img = Image.open(img_file)
    if img.format == 'PNG':
        return SCAN_IMAGE_TYPE

    raw_info = img._getexif()
    if not raw_info:
        return SCAN_IMAGE_TYPE

    return PHOTO_IMAGE_TYPE


def detect_license_image(img_file):
    img = Image.open(img_file)
    ratio = img.width * 1.0 / img.height
    if img.width == 2374: # hard fix 
        return 'certificate'
    if img.width < 4200 and 1.48 < ratio < 1.8:
        return 'license'
    return 'certificate'


def check_color(img_file):
    img = Image.open(img_file)
    # print(img.shape)
    img_array = np.asarray(img, dtype=np.uint8)
    if len(img_array.shape) < 3:
        return False
    n, m, k = img_array.shape
    num_color_points = 0.0
    if n * m < 3000 * 1000:
        for i in range(n):
            for j in range(m):
                if max(img_array[i][j]) - min(img_array[i][j]) > 50:
                    num_color_points += 1
        print(num_color_points / n / m)
        return num_color_points / n / m > 0.3
    u = [0, 1, 2, 4]
    v = [3, 2, 4, 1]
    for i in range(n / 5):
        for j in range(m / 5):
            for k in range(4):
                if max(img_array[5 * i + u[k]][5 * j + v[k]]) - min(img_array[5 * i + u[k]][5 * j + v[k]]) > 30:
                    num_color_points += 1
    print(num_color_points / n / m)
    return num_color_points / n / m > 0.01
