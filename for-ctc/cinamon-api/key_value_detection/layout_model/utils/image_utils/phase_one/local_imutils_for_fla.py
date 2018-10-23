import os
import cv2
import json
from ..bboxs_tools.bbox_operations import (
    map_ratio_to_parent_ref_points as map_ratio_p,
    crop_bbox_to_file,
    scale_bbox_ref_point,
    )
from ....file_utils import make_sure_dir_exists


refPt = []


def click_and_crop(event, x, y, flags, param):
    # grab references to the global variables
    global refPt

    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]

    # check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates and indicate that
        # the cropping operation is finished
        refPt.append((x, y))


def crop_scale_bbox_ref_point(image, ref_pt, ratio, path):
    bounding_box = scale_bbox_ref_point(ref_pt, ratio)
    crop_bbox_to_file(image, bounding_box, path)


def crop_tables_from_mouse_click(image):
    global refPt
    clone = image.copy()
    image = clone.copy()
    ratio = 1.0
    while clone.shape[0] > 768:
        ratio -= 0.01
        clone = cv2.resize(image.copy(),
                           (int(image.shape[1]*ratio),
                            int(image.shape[0]*ratio)))
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", click_and_crop)
    total_dict = {"tables": {}}
    new_table, new_key, new_table_name, new_value = True, True, False, False
    current_table, current_key = 0, -1
    current_table_pos, current_key_pos, current_value_pos = [], [], []
    output_table_dir = make_sure_dir_exists("table_crop")
    output_table_name_dir = make_sure_dir_exists("table_name_crop")
    output_key_dir = make_sure_dir_exists("key_crop")
    while True:
        # display the image and wait for a keypress
        cv2.imshow("image", clone)
        key = cv2.waitKey(1) & 0xFF

        # if the 'r' key is pressed, reset the cropping region
        if key == ord('r'):
            if not new_table:
                current_table += 1
            new_table = True

        # if the 'n' key is pressed, adding table name
        elif key == ord('n'):
            new_table_name = True
        # if the 'k' key is pressed, adding new key region
        elif key == ord('k'):
            if not new_key:
                current_key += 1
            new_table = False
            new_key = True

        # if the 'v' key is pressed, adding new value to key
        elif key == ord('v'):
            new_table = False
            if new_value:
                refPt = []
            new_value = True
        elif key == ord('c') and new_value:
            refPt = current_value_pos
            new_value = False
            cv2.rectangle(clone, refPt[0], refPt[1], (0, 0, 255), 2)
            sub_table_x_range, sub_table_y_range = map_ratio_p(
                current_value_pos, current_table_pos)
            total_dict['tables'][str(current_table)]['keys'][
                str(current_key)].append(
                {
                    'bbox_index': current_table,
                    'sub_table_x_range': sub_table_x_range,
                    'sub_table_y_range': sub_table_y_range
                }
            )
            new_table, new_table_name,\
                new_key, new_value = False, False, False, False
            refPt = []

        # if the 'q' key is pressed, break from the loop
        elif key == ord('q'):
            break

        # if there are two reference points, then draw rect
        if len(refPt) == 2:
            if new_table:
                current_table_pos = refPt.copy()
                crop_scale_bbox_ref_point(
                    image, current_table_pos, 1.0/ratio,
                    os.path.join(output_table_dir, str(current_table)+".jpg")
                )
                total_dict['tables'][str(current_table)] = {"keys": {}}
                cv2.rectangle(clone, refPt[0], refPt[1], (0, 255, 0), 2)
                new_table, new_table_name,\
                    new_key, new_value = False, False, False, False
            elif new_table_name:
                current_table_name_pos = refPt.copy()
                cv2.rectangle(clone, refPt[0], refPt[1], (255, 255, 0), 2)
                crop_scale_bbox_ref_point(
                    image, current_table_name_pos, 1.0/ratio,
                    os.path.join(output_table_name_dir,
                                 str(current_table)+".jpg"))
                new_table, new_table_name,\
                    new_key, new_value = False, False, False, False
            elif new_key:
                current_key_pos = refPt.copy()
                cv2.rectangle(clone, refPt[0], refPt[1], (0, 255, 255), 2)
                crop_scale_bbox_ref_point(
                    image, current_key_pos, 1.0/ratio,
                    os.path.join(output_key_dir, str(current_key)+".jpg"))
                total_dict['tables'][
                    str(current_table)
                ]['keys'][str(current_key)] = []

                new_table, new_table_name,\
                    new_key, new_value = False, False, False, False
            elif new_value:
                current_value_pos = refPt.copy()
            refPt = []
    json.dump(total_dict, open("output_middle.json", "w"), indent=4)


def middle_format_to_real_format(middle_json="output_middle_7.json",
                                 key_name_dir="key_crop_7",
                                 table_name_dir="table_name_7"):
    tables = json.load(open(middle_json, "r"))
    old_table_keys = [table_key for table_key in tables["tables"].keys()]
    for table_key in old_table_keys:
        new_table_name = open(os.path.join(table_name_dir, table_key+".txt"),
                              "r").read()
        tables["tables"][new_table_name] = tables["tables"].pop(table_key)
        old_keys = [old_key
                    for old_key in
                    tables["tables"][new_table_name]['keys'].keys()]
        for old_key in old_keys:
            new_table_key = open(os.path.join(
                key_name_dir, old_key+".txt"),
                                  "r").read()
            tables[
                "tables"
            ][new_table_name
              ]['keys'
                ][new_table_key] = tables[
                    "tables"][new_table_name]['keys'].pop(old_key)
    json.dump(tables, open("output_format_from_middle.json", "w"),
              indent=4, ensure_ascii=False)
