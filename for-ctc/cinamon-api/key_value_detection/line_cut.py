import os, shutil, glob, tempfile, subprocess, re
import unicodedata
from key_value_detection.layout_model.layout_model import LayoutModel
import xml.etree.ElementTree as ET

RESULT_FOLDER = 'result'

def process_one_image(image_file, layout_model):
    """
    Do layout analysis, cell cut, line cut.

    :param image_file:  path of image.
    :param layout_model: layout model.
    :return: data.json, images of line cut in folder result.
    """

    file_basename = os.path.basename(image_file).split('.')[0]
    folder = os.path.dirname(image_file)
    linecut_folder = os.path.join(folder, RESULT_FOLDER + '_' + file_basename)
    if not os.path.exists(linecut_folder):
        os.mkdir(linecut_folder)
    shutil.copyfile(image_file, os.path.join(linecut_folder, '_original.png'))

    # layout analytic and cut cell
    layout_model.process_one_file(image_file, linecut_folder)


def run_ocr(image_folder, model_folder = ''):
    """
    Run tesseract-ocr on all images in image_folder.

    :param image_folder: path of folder that contains line-cut images.
    :param model_folder: path of folder that contains data of tesseract-ocr model.
    :return: a dictionary, key is line_name, value is text:
            text_lines['line_name'] = value
    """

    # check model folder
    if not model_folder:
        raise ValueError('model_folder is none')

    text_lines = {}
    img_list = glob.glob(os.path.join(image_folder, '*.png'))
    line_names = [os.path.basename(img_path).replace('.png', '') for img_path in img_list]
    img_list = '\n'.join(img_list)

    with tempfile.TemporaryDirectory() as tmp:
        img_list_txt = os.path.join(tmp, 'img_list.txt')
        open(img_list_txt, 'w').write(img_list)
        text = subprocess.run(f'tesseract {img_list_txt} stdout -l jpn_best --psm 6 --tessdata-dir '
                              + model_folder, shell=True, stdout=subprocess.PIPE)
    text = text.stdout.decode()
    text = unicodedata.normalize('NFKC', text)
    text = text.replace('\n', '')
    text = text.split('\x0c')
    text = text[:-1]
    for i, value in enumerate(text):
        text_lines[line_names[i]] = value
    return text_lines


def parse_hocr(hocr_file):
    """Read hocr file and parse to dictionary"""

    lines = {}
    class_page = ".//*[@class='ocr_page']"
    class_word = ".//*[@class='ocrx_word']"
    tree = ET.parse(hocr_file)
    page_elements = tree.findall(class_page)
    for page_element in page_elements:
        title = page_element.get('title')
        image_file = re.search(r'/.+\.png', title).group()
        image_name = image_file.split('/')[-1].replace('.png', '')
        text = ''
        confidences = []
        word_elements = page_element.findall(class_word)
        for word_element in word_elements:
            words = word_element.text
            confidence = word_element.get('title').split(';')[1].replace(' x_wconf ', '')
            text = text + words
            for i in range(0, len(words)):
                confidences.append(float(confidence)/100.0)
        lines[image_name] = {'text': text, 'confidences': confidences}
    return lines


def run_hocr(image_folder, model_folder = '/Users/anh/tesseract/tessdata'):
    """ Run tesseract-ocr on all images in image_folder"""

    img_list = glob.glob(os.path.join(image_folder, '*.png'))
    img_list.sort(key=lambda x: x)
    img_list = '\n'.join(img_list)
    output_hocr = os.path.join(image_folder, 'output')

    with tempfile.TemporaryDirectory() as tmp:
        img_list_txt = os.path.join(tmp, 'img_list.txt')
        open(img_list_txt, 'w').write(img_list)
        cmd = 'tesseract {0} {1} -l Japanese --psm 6 --tessdata-dir {2} hocr'
        cmd = cmd.format(img_list_txt, output_hocr, model_folder)
        subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    hocr_file = output_hocr + '.hocr'
    lines = parse_hocr(hocr_file)
    return lines


def main():
    image_file = '/Users/anh/Downloads/test_bprost/00004/00004.png'
    textline_model_path = '/Users/anh/Downloads/test_bprost/model/ff_model_textline.pb'
    border_model_path = '/Users/anh/Downloads/test_bprost/model/ff_model_border.pb'

    # process_one_image(image_file, textline_model_path, border_model_path)


if __name__ == '__main__':
    main()
