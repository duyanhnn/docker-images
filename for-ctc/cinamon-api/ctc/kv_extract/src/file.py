import json
import os
import glob
import subprocess
import io

import xlsxwriter

#PDF2PNG_CMD = 'convert -quality 100 -density 150 {} {}'
PDF2PNG_CMD = 'convert -quality 100 -density 150 -background white -alpha remove -append {} {}'

class ConvertErrorException(Exception):
    pass

def read_json_file(file):
    with open(file) as f:
        data = f.read()
    return json.loads(data)

def dump_to_file(data, file):
    with open(file, 'w') as fp:
        json.dump(data, fp)

def pdf_to_image(pdf_file, image_file):
    command = PDF2PNG_CMD.format(pdf_file, image_file)
    command = command.split()
    exit_code = 0
    output = ''
    try:
        output = subprocess.check_output(command, shell=False, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        exit_code = e.returncode
    if 'error' in str(output).lower():
        raise ConvertErrorException()
    return image_file
    for img_name in glob.glob('{}/{}*.png'.format(directory, fn)):
        if "-0.png" in img_name:
            new_name = '{}/{}.png'.format(directory, fn)
            os.rename(img_name, new_name)
        elif "{}.png".format(fn) in img_name:
            new_name = '{}/{}.png'.format(directory, fn)
    return new_name

def generate_ocr_excel(source_directory):
    ocr_json = os.path.join(source_directory, 'data2.json')
    with open(ocr_json) as f:
        data = f.read()
    data = eval(data)

    max_width = [(v['location'][3] - v['location'][1]) for k, v in data.iteritems() if v['value'].strip()]
    avg_width = sum(max_width) / len(max_width)

    output = io.BytesIO()
    workbook = xlsxwriter.Workbook(output, {'in_memory': True})
    font_size_format = workbook.add_format()
    font_size_format.set_font_size(12)
    font_size_format.set_align('center')
    font_size_format.set_align('vcenter')
    font_size_format.set_align('hcenter')
    font_size_format.set_border()
    font_size_format.set_text_wrap(True)
    worksheet = workbook.add_worksheet()
    worksheet.set_column('A:A', 100)
    worksheet.set_column('B:B', 20)
    worksheet.set_column('C:C', 50)
    worksheet.set_column('D:D', 50)
    worksheet.set_row(0, 25)
    worksheet.write('A1', 'Image', font_size_format)
    worksheet.write('B1', 'Image Name', font_size_format)
    worksheet.write('C1', 'OCR value', font_size_format)
    worksheet.write('D1', 'Right Answer', font_size_format)

    count = 2
    sorted_key = sorted(data.keys(), key=lambda item: int(item.replace('table1_cell', '')))
    for k in sorted_key:
        v = data[k]
        value = v['value'].strip()
        loc = v['location']
        height = loc[2] - loc[0]
        width = loc[3] - loc[1]
        if value:
            image = os.path.join(source_directory, '{}.png'.format(k))
            worksheet.set_row(count-1, height)
            if width > avg_width:
                worksheet.insert_image('A{}'.format(count), image, {'x_scale': 0.8, 'y_scale': 0.8})
            else:
                worksheet.insert_image('A{}'.format(count), image)
            worksheet.write('B{}'.format(count), k, font_size_format)
            worksheet.write('C{}'.format(count), value.decode('utf-8'), font_size_format)
            worksheet.write('D{}'.format(count), '', font_size_format)
            count += 1
    for k in sorted_key:
        v = data[k]
        value = v['value'].strip()
        loc = v['location']
        height = loc[2] - loc[0]
        width = loc[3] - loc[1]
        if not value:
            image = os.path.join(source_directory, '{}.png'.format(k))
            worksheet.set_row(count-1, height)
            if width > avg_width:
                worksheet.insert_image('A{}'.format(count), image, {'x_scale': 0.8, 'y_scale': 0.8})
            else:
                worksheet.insert_image('A{}'.format(count), image)
            worksheet.write('B{}'.format(count), k, font_size_format)
            worksheet.write('C{}'.format(count), '', font_size_format)
            worksheet.write('D{}'.format(count), '', font_size_format)
            count += 1
    workbook.close()
    output.seek(0)
    return output
