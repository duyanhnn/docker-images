from scipy.ndimage.measurements import label as cc_label
from scipy.ndimage.measurements import find_objects
import cv2
import os
import json


def get_connected_components(image, thres = 0, reverse = True):
    """Set image to binary and find connectec components"""

    if thres > 0:
        binary = image > thres
    else:
        binary = image

    if reverse == True:
        binary = 1-binary

    labels, _ = cc_label(binary)
    objects = find_objects(labels)
    return labels, objects

def draw_data_cell(image_file, data_json_file, new_image_name = 'cell.png'):
    """Draw boxes in json file into image"""

    color = [(255, 0, 0),(0, 0, 255)]
    folder = os.path.dirname(os.path.realpath(image_file))

    # load location data
    with open(data_json_file) as f:
        data = json.load(f)

    idx = 0
    image = cv2.imread(image_file)
    for k, v in data.items():
        if 'line' in k:
            continue
        y1, x1, y2, x2 = v['location']
        idx = 1 if idx==0 else 0
        cv2.rectangle(image, (x1, y1), (x2, y2), color[idx], 3)
        cv2.putText(image, k, (x1+5, y1+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.imwrite(os.path.join(folder, new_image_name), image)


def draw_data_json(image_file, data_json_file, color = None, new_image_name = 'test.png'):
    """Draw boxes in json file into image"""

    color = (255, 0, 0) if color is None else color
    folder = os.path.dirname(os.path.realpath(image_file))

    # load location data
    with open(data_json_file) as f:
        data = json.load(f)

    image = cv2.imread(image_file)
    for k, v in data.items():
        if 'line' not in k:
            continue
        y1, x1, y2, x2 = v['location']
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
    cv2.imwrite(os.path.join(folder, new_image_name), image)