import sys
sys.path.append('core')

import os
#from ocr import ocr
from glob import glob
import json
#from core.model import LineCutModel

IMG_DIR = 'data/input_v1'
OCR_DIR = 'data/ocr_v1'
LINECUT_MODEL_JSON = 'models/linecut_model.json'
LINECUT_MODEL_WEIGHT = 'models/linecut_model_weight.h5'

class Preprocessor():
    
    def __init__(self, model_json=LINECUT_MODEL_JSON, model_weight=LINECUT_MODEL_WEIGHT):
        self.linecut_model = LineCutModel()
        self.linecut_model.load(model_json=model_json, model_weight=model_weight)
        self.img_dir = sorted(glob(os.path.join(IMG_DIR, '*.png')))

    def ocr(self, img_path):
        lines, _, debug_img = self.linecut_model.predict(img_path)
        out, lines = ocr(img_path, lines)
        return out, lines

    def preprocess(self):
        '''Prepare data for entity model training'''
        for img_path in self.img_dir:
            out, lines = self.ocr(img_path)
            img_name = os.path.basename(img_path)
            img_name = img_name.split('.')[0]

            if not os.path.exists(OCR_DIR):
                os.mkdir(OCR_DIR)

            ocr_path = os.path.join(OCR_DIR, img_name + '.txt')
            open(ocr_path, 'x').write(out)

            doubled_path = os.path.join(OCR_DIR, 'doubled')
            if not os.path.exists(doubled_path):
                os.mkdir(doubled_path)
            out_doubled = double_lines(out)
            ocr_path = os.path.join(doubled_path, img_name + '.txt')
            open(ocr_path, 'x').write(out_doubled)

            lines = serialize_lines(lines)
            box_path = os.path.join(OCR_DIR, img_name + '.csv')
            open(box_path, 'x').write(lines)

def serialize_lines(lines):
    lines = [','.join(map(str, line)) for line in lines]
    lines = '\n'.join(lines)
    return lines

def double_lines(text):
    '''Double each line in given text str for easier labeling'''
    lines = []
    for line in text.splitlines():
        lines.append(line)
        lines.append(line)
    lines = '\n'.join(lines)
    return lines

def parse_line_clf_labels(path):
    '''Parse line labels in doubled line format
    Args:
        path: path to line label file
    Returns:
        result: dict mapping from line text to line class'''
    result = {}
    code_to_label_dict = {
            'c': 'company',
            'a': 'address',
            'd': 'address'
            }
    lines = open(path, 'r').read().splitlines()
    text_lines = lines[::2]
    label_lines = lines[1::2]
    for text_line, label_line in zip(text_lines, label_lines):
        if text_line == label_line:
            label = 'null'
        else:
            for code in label_line.split(','):
                try:
                    label = code_to_label_dict[code]
                except KeyError:
                    label = 'null'
        result[text_line] = label
    return result

if __name__ == '__main__':
    LINE_CLF_LABEL_PATH_LIST = [os.path.join('data', path.strip('\n')) for path in open('data/ocr_v3/done.txt', 'r')]
    result = {}
    for path in LINE_CLF_LABEL_PATH_LIST:
        result.update(parse_line_clf_labels(path))
    json.dump(result, open('data/line_clf_data_ocr_v3.json', 'w'), ensure_ascii=False, indent=4, sort_keys=True)
    print(result)
