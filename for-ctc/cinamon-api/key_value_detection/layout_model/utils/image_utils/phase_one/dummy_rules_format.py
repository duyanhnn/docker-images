import os
import re
import cv2
import xlsxwriter
from cv2_utils import draw_cover_rectangle
from ..bboxs_tools.bbox_operations import (
    get_area_bounding_box,
    get_x_y_start_stop,
    get_aspect_ratio_bounding_box,
    get_boundingbox_tl_lr,
    sort_bbox_top_down, sort_bbox_bottom_up,
    sort_bbox_left_to_right,
    cut_bbox_take_right, cut_bbox_take_left,
    crop_bbox, crop_bbox_to_file,
    cut_bbox_take_range_x_y
    )

from ..local_imutils import (
    threshold_on_max,
    invert_color_image,
    connected_component_analysis_sk as cca_sk,
    tesseract4_wrap_all_image as tesseract_wrap_all_image
    )

import json

DEBUGGING = True
if DEBUGGING:
    from pylab import imshow, show


def dummy_table_format_2(gray_image):
    invert_binarized_image = threshold_on_max(
        invert_color_image(gray_image.copy()),
        0.2)
    rects2, rects_types, lines, lines_y_range, big_areas = cca_sk(
        invert_binarized_image)
    # Filter out vertical boxes and too long boxes
    big_areas = [big_area for big_area in big_areas
                 if get_aspect_ratio_bounding_box(big_area) > 1.0]
    # Find NINE largest areas
    areas = [get_area_bounding_box(big_area) for big_area in big_areas]
    sorted_indices_big_areas = [
        i[0] for i in sorted(enumerate(areas), key=lambda x:x[1])][::-1]
    big_areas = [big_areas[i] for i in sorted_indices_big_areas[:9]]
    # Sort bboxs top down, takes the 8 lower
    big_areas = sort_bbox_top_down(big_areas)[0][1:]
    big_areas_non_blackbox = big_areas
    print(len(big_areas_non_blackbox))
    return big_areas_non_blackbox


def dummy_table_format_4(gray_image):
    image_bbox = get_boundingbox_tl_lr(
        (0, 0), (gray_image.shape[0], gray_image.shape[1]))
    left_page_bbox = cut_bbox_take_left(image_bbox, 0.5)
    right_page_bbox = cut_bbox_take_right(image_bbox, 0.5)
    left_page = crop_bbox(gray_image.copy(), left_page_bbox)
    right_page = crop_bbox(gray_image.copy(), right_page_bbox)
    invert_binarized_left = threshold_on_max(
        invert_color_image(left_page.copy()),
        0.2)
    invert_binarized_right = threshold_on_max(
        invert_color_image(right_page.copy()),
        0.1)
    if DEBUGGING:
        imshow(invert_binarized_right)
        show()
    _, _, _, _, big_areas_left = cca_sk(
        invert_binarized_left)
    # Find the BIGGEST area
    areas_left = [get_area_bounding_box(big_area)
                  for big_area in big_areas_left]
    sorted_indices_big_areas = reversed(
        [i[0] for i in sorted(enumerate(areas_left), key=lambda x:x[1])])
    table_lefts = [big_areas_left[i] for i in sorted_indices_big_areas[:1]]
    return table_lefts


def dummy_table_format_7(gray_image):
    invert_binarized_image = threshold_on_max(
        invert_color_image(gray_image.copy()),
        0.4)
    rects2, rects_types, lines, lines_y_range, big_areas = cca_sk(
        invert_binarized_image)
    # Filter out too long boxes
    big_areas = [big_area for big_area in big_areas
                 if get_aspect_ratio_bounding_box(big_area) < 30]
    # Sort bboxs bottom up
    big_areas = sort_bbox_bottom_up(big_areas)[0][:15]
    # Sort top down
    big_areas = sort_bbox_top_down(big_areas)[0]
    big_areas_top = sort_bbox_left_to_right(big_areas[:3])[0]
    print(len(big_areas))
    big_areas = big_areas[3:]
    print(len(big_areas))
    _, x_start, y_start, y_stop = get_x_y_start_stop(big_areas_top[0])
    x_stop, _, _, _ = get_x_y_start_stop(big_areas_top[1])
    first_boundary = get_boundingbox_tl_lr((y_start, x_start), (y_stop,
                                                                x_stop-30))
    new_big_areas = [first_boundary]
    for big_area in big_areas[:7]:
        _, x_start, y_start, y_stop = get_x_y_start_stop(big_area)
        x_stop = gray_image.shape[1]
        new_big_areas.append(get_boundingbox_tl_lr((y_start-10, x_start),
                                                   (y_stop, x_stop-30)))

    print(len(big_areas))
    big_areas_bottom = sort_bbox_left_to_right(big_areas[7:])[0]
    divider = big_areas_bottom.pop(2)
    big_areas_bottom[:2] = sort_bbox_top_down(big_areas_bottom[:2])[0]
    big_areas_bottom[2:] = sort_bbox_top_down(big_areas_bottom[2:])[0]
    print("Big_areas_bottom: ", len(big_areas_bottom))
    new_big_areas.append(cut_bbox_take_right(big_areas_bottom[0], 0.633))
    _, x_start, y_start, y_stop = get_x_y_start_stop(big_areas_bottom[1])
    x_stop, _, _, _ = get_x_y_start_stop(divider)
    new_big_areas.append(get_boundingbox_tl_lr((y_start, x_start),
                                               (y_stop, x_stop-30)))

    for bbox in big_areas_bottom[2:]:
        new_big_areas.append(cut_bbox_take_right(bbox, 0.07))
    return new_big_areas


def dummy_full_format_2(gray_image, format_json_file='2.format.json',
                        format_name='2',
                        dummy_table_function=dummy_table_format_2):
    big_areas = dummy_table_function(gray_image)
    tables = json.load(open(format_json_file))['tables']
    tables_out = {}
    out_dir_img = "format" + format_name + "_cells_images"
    if not os.path.exists(out_dir_img):
        os.makedirs(out_dir_img)
    out_dir_text = "format" + format_name + "_cells_texts"
    if not os.path.exists(out_dir_text):
        os.makedirs(out_dir_text)
    k = 0
    bboxs = []
    workbook = xlsxwriter.Workbook('format'+format_name+'_output.xlsx')
    for table_name in tables.keys():
        worksheet = workbook.add_worksheet(table_name[:31])
        i = 0
        tables_out[table_name] = {}
        table = tables[table_name]
        for table_key in table['keys'].keys():
            worksheet.write(i, 0, table_key)
            j = 1
            tables_out[table_name][table_key] = []
            for table_value_cut in table['keys'][table_key]:
                print(
                    "k: %d, bbox_index: %d, x_s: %f, x_e: %f,\
                    y_s: %f, y_e: %f" % (
                        k,
                        table_value_cut['bbox_index'],
                        table_value_cut['sub_table_x_range'][0],
                        table_value_cut['sub_table_x_range'][1],
                        table_value_cut['sub_table_y_range'][0],
                        table_value_cut['sub_table_y_range'][1])
                )
                bbox = cut_bbox_take_range_x_y(
                                       big_areas[table_value_cut['bbox_index']],
                                       table_value_cut['sub_table_x_range'][0],
                                       table_value_cut['sub_table_x_range'][1],
                                       table_value_cut['sub_table_y_range'][0],
                                       table_value_cut['sub_table_y_range'][1])

                bboxs.append(bbox)
                crop_bbox_to_file(gray_image, bbox,
                                  os.path.join(out_dir_img, str(k)+".jpg")
                                  )
                cell_text = tesseract_wrap_all_image(
                                   os.path.join(out_dir_img, str(k)+".jpg"),
                                   os.path.join(out_dir_text, str(k))
                                   )
                worksheet.write(i, j, cell_text)

                tables_out[table_name][table_key].append(cell_text)
                j += 1
                k += 1
            i += 1
    draw_cover_rectangle(gray_image, bboxs, "dummy_"+format_name+"_all.jpg")
    json.dump(tables_out, open("output_" + format_name + ".json", "w"),
              ensure_ascii=False, indent=4)
    workbook.close()


def dummy_full_format_3(num_tables=6, output_file="format3_output",
                        format_file="3.format.json", table_image_dir="gray1",
                        plus=1, the_format=".PNG", offset=0):
    big_areas_image = [cv2.imread(
        os.path.join(table_image_dir, str(int(i+plus)) + the_format))
                       for i in range(num_tables)
                       ]
    big_areas = [
        get_boundingbox_tl_lr((0, 0), (big_area.shape[0], big_area.shape[1]))
        for big_area in big_areas_image
    ]
    bboxs = []
    tables = json.load(open(format_file))['tables']
    tables_out = {}
    out_dir_img = output_file+"_image"
    if not os.path.exists(out_dir_img):
        os.makedirs(out_dir_img)
    out_dir_text = output_file+"_text"
    if not os.path.exists(out_dir_text):
        os.makedirs(out_dir_text)
    k = 0
    workbook = xlsxwriter.Workbook(output_file+'.xlsx')
    for table_name in tables.keys():
        worksheet = workbook.add_worksheet(re.sub(":", "",
                                                  table_name[:31]))
        i = 0
        tables_out[table_name] = {}
        table = tables[table_name]
        for table_key in table['keys'].keys():
            worksheet.write(i, 0, table_key)
            j = 1
            tables_out[table_name][table_key] = []
            for table_value_cut in table['keys'][table_key]:
                print(
                    "k: %d, bbox_index: %d, x_s: %f, x_e: %f,\
                    y_s: %f, y_e: %f" % (
                        k,
                        table_value_cut['bbox_index']+offset,
                        table_value_cut['sub_table_x_range'][0],
                        table_value_cut['sub_table_x_range'][1],
                        table_value_cut['sub_table_y_range'][0],
                        table_value_cut['sub_table_y_range'][1])
                )
                bbox = cut_bbox_take_range_x_y(
                    big_areas[table_value_cut['bbox_index']+offset],
                    table_value_cut['sub_table_x_range'][0],
                    table_value_cut['sub_table_x_range'][1],
                    table_value_cut['sub_table_y_range'][0],
                    table_value_cut['sub_table_y_range'][1])

                bboxs.append(bbox)
                gray_image = big_areas_image[
                    table_value_cut['bbox_index']+offset]
                crop_bbox_to_file(gray_image, bbox, os.path.join(
                    out_dir_img, str(k)+".jpg"))
                cell_text = tesseract_wrap_all_image(
                                   os.path.join(out_dir_img, str(k)+".jpg"),
                                   os.path.join(out_dir_text, str(k))
                                   )
                worksheet.write(i, j, cell_text)

                tables_out[table_name][table_key].append(cell_text)
                j += 1
                k += 1
            i += 1
    json.dump(tables_out, open(output_file+".json", "w"),
              ensure_ascii=False, indent=4)
    workbook.close()


def dummy_full_format_4():
    dummy_full_format_3(8, output_file="format4_output",
                        format_file="4.format.json",
                        table_image_dir="table_crop_4",
                        plus=0, the_format=".jpg", offset=-1)


def dummy_full_format_5():
    dummy_full_format_3(5, output_file="format5_output",
                        format_file="5.format.json",
                        table_image_dir="table_crop_5",
                        plus=0, the_format=".jpg")


def dummy_full_format_6():
    dummy_full_format_3(4, output_file="format6_output",
                        format_file="6.format.json", table_image_dir="tables_6",
                        plus=0)


def dummy_full_format_7(gray_image):
    dummy_full_format_2(gray_image, format_json_file='7.format.json',
                        format_name='7',
                        dummy_table_function=dummy_table_format_7)


def dummy_full_format_8():
    dummy_full_format_3(6, output_file="format8_output",
                        format_file="8.format.json", table_image_dir="tables_8",
                        plus=1)


def dummy_full_format_9():
    dummy_full_format_3(6, output_file="format9_output",
                        format_file="9.format.json",
                        table_image_dir="table_crop_9",
                        plus=0, the_format=".jpg")


def dummy_full_format_12():
    dummy_full_format_3(5, output_file="format12_output",
                        format_file="12.format.json",
                        table_image_dir="table_crop_12",
                        plus=0, the_format=".jpg")
