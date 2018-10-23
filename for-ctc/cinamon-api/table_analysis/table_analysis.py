import json
import cv2
import os
import time
import numpy as np
from table_util import get_connected_components, draw_data_cell


def fill_cell_and_dilation(mask, data, kernel = None):
    """
    Open cells and fill cells with 1

    :param mask: mask with all 0
    :param data: cell's location
    :param kernel: (width, height) to open cells
    :return: mask - background 0 and foregound = 1
    """

    if kernel is None:
        kernel = (15,15)

    # fill one all cells
    for k, v in data.items():
        y1, x1, y2, x2 = v['location']
        x1, y1 = max(x1-kernel[0],0), max(y1 - kernel[1],0)
        x2, y2 = min(x2 + kernel[0], mask.shape[1]), min(y2 + kernel[1], mask.shape[0])
        mask[y1:y2 + 1, x1:x2 + 1] = 1
    return mask


def find_cell_in_table(tables, data):
    """
    Determind cell in table

    :param tables: each table is a connected component
    :param data: contain location of cell
    :return:
        cells_in_table: dictionary of table's cell info, every table contains a list of cell's name
        location_table: dictionary of table's info, contain location of table
    """

    cells_in_table = {}
    location_table = {}
    for k, v in data.items():
        y1, x1, y2, x2 = v['location']
        for i, table in enumerate(tables):
            table_y1, table_y2 = table[0].start, table[0].stop
            table_x1, table_x2 = table[1].start, table[1].stop
            if y1 >= table_y1 and y2 <= table_y2 and x1 >= table_x1 and x2 < table_x2:
                if 'table' + str(i+1) not in cells_in_table:
                    cells_in_table['table' + str(i+1)] = [k]
                    location_table['table' + str(i+1)] = [x1, y1, x2, y2]
                    break
                else:
                    cells_in_table['table' + str(i+1)].append(k)
                    x1_tmp, y1_tmp, x2_tmp, y2_tmp = location_table['table' + str(i + 1)]
                    x1_tmp, y1_tmp = min(x1_tmp, x1), min(y1_tmp, y1)
                    x2_tmp, y2_tmp = max(x2_tmp, x2), max(y2_tmp, y2)
                    location_table['table' + str(i + 1)] = [x1_tmp, y1_tmp, x2_tmp, y2_tmp]
                    break
    return cells_in_table, location_table


def get_cell_information(cells_in_table, location_table, data):
    """
    Get postion of cell in table as row, column
    Get all cells and structure of table

    :param cells_in_table: dictionary of table's cell info, every table contains a list of cell's name
    :param location_table: dictionary of table's info, contain location of table
    :param data: contain location of cell
    :return:
        cells: postions of cell in table
            {
                'cell1': {'table_name': 'table_1', 'position':[(row,column),...]},
                ...
            }

        tables: all cells and structure of table
            {
                'table_1': [['cell1', 'cell2', 'cell3', 'cell4'],
                            ['cell5', 'cell6', 'cell7', 'cell8']]
            }
    """

    cells = {}
    tables = {}
    for table_name in location_table:
        tables[table_name] = []
        table_x1, table_y1, table_x2, table_y2 = location_table[table_name]
        mask = np.ones((table_y2-table_y1+1, table_x2-table_x1+1), np.uint8)
        for cell_name in cells_in_table[table_name]:
            cell_y1, cell_x1, cell_y2, cell_x2 = data[cell_name]['location']
            cell_x1, cell_y1 = cell_x1 - table_x1, cell_y1 - table_y1
            cell_x2, cell_y2 = cell_x2 - table_x1, cell_y2 - table_y1
            mask[cell_y1:cell_y2+1, cell_x1:cell_x2+1] = 0

        # remove columns
        kernel_row = np.ones((1, 50), np.uint8)
        mask_row = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_row)
        kernel_column = np.ones((50, 1), np.uint8)
        mask_column = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_column)

        labels, objects = get_connected_components(mask_row, reverse=False)
        for o in objects:
            obj_x1, obj_y1, obj_x2, obj_y2 = o[1].start, o[0].start, o[1].stop, o[0].stop
            width, height = obj_x2 - obj_x1, obj_y2 - obj_y1
            if width > 2 * height and width < mask.shape[1]:
                mask[obj_y1:obj_y2, 0:mask.shape[1]] = 1

        # remove rows
        labels, objects = get_connected_components(mask_column, reverse=False)
        for o in objects:
            obj_x1, obj_y1, obj_x2, obj_y2 = o[1].start, o[0].start, o[1].stop, o[0].stop
            width, height = obj_x2 - obj_x1, obj_y2 - obj_y1
            if height > 2 * width and height < mask.shape[0]:
                mask[0:mask.shape[0], obj_x1:obj_x2] = 1
        labels, objects = get_connected_components(mask, reverse=True)

        # get info of cell
        cell_list = []
        for i, o in enumerate(objects):
            obj_x1, obj_y1, obj_x2, obj_y2 = o[1].start, o[0].start, o[1].stop, o[0].stop
            center = (int((obj_x2 - obj_x1) / 2 + obj_x1), int((obj_y2 - obj_y1) / 2 + obj_y1))

            if i == 0:
                row, col = 0, 0
                cell_list = []
            else:
                if objects[i-1][1].stop < obj_x1:
                    col += 1
                else:
                    col = 0
                    row += 1
                    cell_list = []

            for cell_name in cells_in_table[table_name]:
                cell_y1, cell_x1, cell_y2, cell_x2 = data[cell_name]['location']
                cell_x1, cell_y1 = cell_x1 - table_x1, cell_y1 - table_y1
                cell_x2, cell_y2 = cell_x2 - table_x1, cell_y2 - table_y1
                # check if new cell in cell
                if cell_x1 < center[0] and center[0] < cell_x2 \
                        and cell_y1 < center[1] and center[1] < cell_y2:
                    # add cell
                    if cell_name not in cells:
                        cells[cell_name] = {'table_name': table_name, 'position': [(row, col)]}
                    else:
                        cells[cell_name]['position'].append((row, col))
                    cell_list.append(cell_name)
                    break
            if i == len(objects) - 1 or objects[i+1][1].start < obj_x2:
                tables[table_name].append(cell_list)

    return cells, tables


def get_table_cell_information(data_json_file, image_file):

    # load location data
    data = {}
    with open(data_json_file) as f:
        json_data = json.load(f)
    for k, v in json_data.items():
        if 'line' not in k:
            data[k] = v

    image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)

    # create mask with all's 0
    mask = np.zeros(image.shape, np.uint8)

    # open cells and fill cells with 1
    mask = fill_cell_and_dilation(mask, data)

    # get connected component
    labels, objects = get_connected_components(mask, reverse=False)

    # determind cell in table
    cells_in_table, location_table = find_cell_in_table(objects, data)

    # get information of cell and table
    cells, tables = get_cell_information(cells_in_table, location_table, data)
    return cells, tables


if __name__ == '__main__':
    start = time.time()
    data_json_file = '/Users/anh/Downloads/test_bprost/00004/result/data.json'
    image_file = '/Users/anh/Downloads/test_bprost/00004/00004.png'
    # draw_data_cell(image_file, data_json_file)
    cells, tables = get_table_cell_information(data_json_file, image_file)
    print('{} s'.format(time.time() - start))
    print(cells)
    print(tables)

