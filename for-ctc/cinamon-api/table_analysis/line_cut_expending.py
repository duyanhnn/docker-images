import json
import cv2
import os
import time
import numpy as np
from table_analysis.table_util import get_connected_components, draw_data_json


def read_image_with_closing(image_file):
    """
    Read image and apply closing to remove noise

    :param image_file: absolute path of image
    :return image: ndarray of image shape 2d
    """

    kernel = np.ones((4, 4), np.uint8)
    image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    return image


def get_area_of_box(labels, box):
    """Get area on label base on size of box"""

    y1, x1, y2, x2 = box
    return labels[y1:y2 + 1, x1:x2 + 1]


def expend_box(box, objects):
    """Expend box base on size of objects"""

    y_min, x_min, y_max, x_max = box
    for obj in objects:
        y_min, y_max = min(y_min, obj[0].start), max(y_max, obj[0].stop)
        x_min, x_max = min(x_min, obj[1].start), max(x_max, obj[1].stop)
    return [y_min-2, x_min-2, y_max+2, x_max+2]


def compute_area(x1, y1, x2, y2):
    """Calculate are of box"""

    area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    return area


def remove_sub_box(dict_location):
    """Find sub_boxes which are in boxes completely and remove them"""

    new_dict = {}
    boxes = list(dict_location.items())
    for box1 in boxes:
        is_child = False
        box1_y1, box1_x1, box1_y2, box1_x2 = box1[1]['location']
        for box2 in boxes:
            if box1 == box2: continue
            box2_y1, box2_x1, box2_y2, box2_x2 = box2[1]['location']
            if box1_x1 >= box2_x1 and box1_y1 >= box2_y1 and box1_x2 <= box2_x2 and box1_y2 <= box2_y2:
                is_child = True
                break
        if not is_child:
            new_dict[box1[0]] = box1[1]
    return new_dict


def expend_line_cut(data_json_file, fix_data_json_file, image_file, debug=False):
    """
    Use conencted component to expend boxes

    :param data_json_file: absolute path of file which contains boxes of line cut step
    :param image_file: absolute path of image
    :param debug: if true, print run time and save image with expended boxes
    :return data_new.json: contain boxes of [y1, x1, y2, x2]
    """

    start_time = time.time()
    # load location data
    data = {}
    with open(data_json_file) as f:
        json_data = json.load(f)
    for k, v in json_data.items():
        if 'line' in k:
            data[k] = v

    # find connected components
    image = read_image_with_closing(image_file)
    image_width = image.shape[1]
    image_height = image.shape[0]
    labels, objects = get_connected_components(image, thres=127)

    count = 1
    dict_location = {}
    for k, v in data.items():
        fist_time = time.time()
        # get mask for each box
        box = v['location']
        area = get_area_of_box(labels, box)

        # find components intersect with box
        ids = np.unique(area)
        obj_ids = (np.delete(ids, 0)-1).astype('int')
        cpn_objects = [objects[id] for id in obj_ids]
        new_box = expend_box(box, cpn_objects)
        if new_box[0] < 0: new_box[0] = 0
        if new_box[1] < 0: new_box[1] = 0
        if new_box[2] > image.shape[0]: new_box[2] = image.shape[0]
        if new_box[3] > image.shape[1]: new_box[3] = image.shape[1]

        # check area condition
        box_area = compute_area(*box)
        new_box_area = compute_area(*new_box)
        if new_box_area > 2 * box_area:
            new_box = box
        dict_location[k] = {'location': new_box}

        if debug == True:
            duration = (time.time() - fist_time)
            print('item {}: {} with {} s'.format(count, k, duration))
            count += 1

    # remove sub_boxes
    locations = remove_sub_box(dict_location)

    # save new data json
    with open(fix_data_json_file, 'w') as f:
        json.dump(locations, f)

    if debug == True:
        end_time = time.time()
        print('Total time: {}'.format(time.time()-start_time))
        draw_data_json(image_file, fix_data_json_file, new_image_name='debug.png')
    return locations,(image_width, image_height)


def main():
    data_json_file = '/Users/anh/Downloads/test_bprost/00004/result/data.json'
    fix_data_json_file = '/Users/anh/Downloads/test_bprost/00004/result/new_data.json'
    image_file = '/Users/anh/Downloads/test_bprost/00004/00004.png'

    # expend boxes of step line cut
    expend_line_cut(data_json_file, fix_data_json_file, image_file, debug=True)

    # # draw image with boxes of step line cut
    # draw_data_json(image_file, data_json_file, color = None, new_image_name = 'test.png')


if __name__ == '__main__':
    main()
