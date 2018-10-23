# from glob import glob
# from functools import reduce
# from skimage import measure, transform
# from skimage.io import imread, imsave
# from operator import itemgetter, add
# from xml.etree import ElementTree
# from models.classify import InferenceModel
# from unicodedata import normalize
# from pprint import pprint
# import os
# import cv2
# from key_value_detection.key_value_detection import detect_fields   # Detects sum, total, tax, bank, branch,
# # account number, account type
# from key_value_detection.layout_model.layout_model import LayoutModel
# import time
# import json
# import difflib
# import tempfile
# import warnings
# import subprocess
# import editdistance
# import numpy as np
# import regex as re
# from wand.image import Image, Color
# from scipy.ndimage.measurements import find_objects
# from scipy.ndimage.measurements import label as cc_label
# from django.conf import settings
#
#
# warnings.filterwarnings("ignore")
#
# LANG = 'jpn_best'
# PSM = '11'
# MIN_AREA_RELATIVE_THRESHOLD = 2.87e-6 # Tuned based on 00002.png
# MAX_AREA_RELATIVE_THRESHOLD = 2.87e-4
# cur_file_path = os.path.abspath(__file__)
#
# LINE_CLASSIFICATION_MODEL_PATH = os.path.dirname(cur_file_path) + os.path.sep + 'models/line_classification/model.h5'
# LINE_CLASSIFICATION_VOCAB_PATH = os.path.dirname(cur_file_path) + os.path.sep + 'models/line_classification/vocab.pkl'
# LINE_CLASSIFICATION_TARGET_PATH = os.path.dirname(cur_file_path) + os.path.sep + 'models/line_classification/target_dict.json'
# COMPANY_DATABASE_PATH = os.path.dirname(cur_file_path) + os.path.sep + 'models/autocorrect/company.json'
# ACCOUNT_DATABASE_PATH = os.path.dirname(cur_file_path) + os.path.sep + 'models/autocorrect/account.json'
#
# TESSDATA_PATH = os.path.dirname(cur_file_path) + os.path.sep + 'models/tessdata'
# LAYOUT_MODEL_PATH = os.path.dirname(cur_file_path) + os.path.sep + 'models/layout/layout.pb'
# BORDER_MODEL_PATH = os.path.dirname(cur_file_path) + os.path.sep + 'models/layout/border.pb'
# BANK_BRANCH_MASTER_PATH = os.path.dirname(cur_file_path) + os.path.sep + 'models/autocorrect/bank_branch_master.csv'
#
# # Persist models between invocations
#
# company_database = json.load(open(COMPANY_DATABASE_PATH))
# company_list = company_database.keys()
# account_database = json.load(open(ACCOUNT_DATABASE_PATH)) # Maps from company names to account names
# account_list = account_database.values()
#
#
# line_classification_model = InferenceModel(
#         LINE_CLASSIFICATION_MODEL_PATH,
#         LINE_CLASSIFICATION_VOCAB_PATH,
#         LINE_CLASSIFICATION_TARGET_PATH
#         )
#
# layout_model = LayoutModel(
#         path_to_pb=LAYOUT_MODEL_PATH,
#         scale=0.35,
#         mode='L',
#         list_path_to_pb_alt=[BORDER_MODEL_PATH]
#         )
#
# class Document():
#     """Object to represent a document"""
#     def __init__(self, lines):
#         lines.sort(key=lambda x: (x['box'][1], x['box'][0]))
#         for line in lines:
#             line['box'] = tuple(line['box'])
#         self.idx2line = dict()
#         self.line2text = {line['box']: line['line'] for line in lines}
#         self.line2line_id = {line['box']: i for i, line in enumerate(lines)}
#         self.text = ''
#         self.width = lines[0]['page_width']
#         self.height = lines[1]['page_height']
#         self.lines = [line['line'] for line in lines]
#         self.boxes = [line['box'] for line in lines]
#         self.confidence_by_char = [line['confidence_by_char'] for line in lines]
#         self.confidence_by_field = [line['confidence_by_field'] for line in lines]
#         for line_id, line in enumerate(lines):
#             line_bbox = line['box']
#             line_text = ''
#             self.line2text[line_bbox] = ''
#             self.line2line_id[line_bbox] = line_id
#             line_confidence_by_char = []
#             for char in line['line']:
#                 line_bbox = line['box']
#                 self.line2text[line_bbox] += char
#                 start = len(self.text)
#                 self.text += char
#                 stop = len(self.text)
#                 for i in range(start, stop):
#                     self.idx2line[i] = line_bbox
#         self._make_line_features()
#
#     def get_line(self, idx):
#         '''Get text belonging to the same line'''
#         line_bbox = self.idx2line[idx]
#         return self.line2text[line_bbox]
#
#     def get_line_id(self, idx):
#         '''Get line id from char id'''
#         line_bbox = self.idx2line[idx]
#         return self.line2line_id[line_bbox]
#
#     def _make_line_features(self):
#         line_features_list = []
#         for line, box, confidence_by_char, confidence_by_field in zip(
#                 self.lines, self.boxes, self.confidence_by_char, self.confidence_by_field):
#             xmin, ymin, xmax, ymax = box
#             line_features = {
#                     'company_id': '',
#                     'line': line,
#                     'corrected_line': line,
#                     'xmin': xmin,
#                     'ymin': ymin,
#                     'xmax': xmax,
#                     'ymax': ymax,
#                     'page_width': self.width,
#                     'page_height': self.height,
#                     'confidence_by_char': confidence_by_char,
#                     'confidence_by_field': confidence_by_field
#                     }
#             line_features_list.append(line_features)
#         self.line_features_list = line_features_list
#
#     def __iter__(self):
#         return iter(self.line_features_list)
#
#
# def read_tif_image(path):
#     '''Handle weird compressed tifs'''
#     with tempfile.TemporaryDirectory() as tmp_dir:
#         out_path = os.path.join(tmp_dir, 'out.png')
#         result = subprocess.run(f'convert {path} {out_path}', shell=True, stdout=subprocess.PIPE)
#         image = imread(out_path)
#     return image
#
#
# def read_pdf_image(pdf_file_path, output_img=None):
#     """Convert pdf to image png"""
#     out_path = output_img
#     with Image(filename=pdf_file_path, resolution=300) as pdf:
#         pages = len(pdf.sequence)
#         if pages > 1:
#             image = Image(
#                 width=pdf.width,
#                 height=pdf.height * pages
#             )
#             for i in range(pages):
#                 image.composite(
#                     pdf.sequence[i],
#                     top=pdf.height * i,
#                     left=0
#                 )
#         else:
#             image = pdf
#         image.background_color = Color("white")
#         image.alpha_channel = 'remove'
#         image.save(filename=out_path)
#         new_image = cv2.imread(out_path)
#     return new_image
#
#
# def remove_red_color(image):
#     """
#     Remove red color space in stamp
#
#     :param image: image 3d [w,h,channel]
#     :return:
#     output_img: image 3d [w,h,channel] which was removed red color space in stamp
#     stamp_location: location of stamp in image [x1,y1,x2,y2]
#     """
#
#     img = image
#     img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#
#     lower_red = np.array([0, 50, 50])
#     upper_red = np.array([10, 255, 255])
#     mask2 = cv2.inRange(img_hsv, lower_red, upper_red)
#
#     lower_red = np.array([170, 50, 50])
#     upper_red = np.array([180, 255, 255])
#     mask4 = cv2.inRange(img_hsv, lower_red, upper_red)
#
#     output_img = img.copy()
#
#     # join masks
#     mask = mask2  + mask4
#     mask_final = np.zeros(mask.shape)
#     # choose areas by condition limit of width, height, ratio
#     kernel = np.ones((10, 10), np.uint8)
#     dilation = cv2.dilate(mask, kernel, iterations=1)
#     labels, _ = cc_label(dilation > 0)
#     objects = find_objects(labels)
#     boxes = []
#     for obj in objects:
#         width = obj[1].stop - obj[1].start
#         height = obj[0].stop - obj[0].start
#         ratio = width/height
#         if ratio < 1:
#             ratio = height/width
#         if 150 < width and 150 < height and ratio < 1.15:
#             mask_final[obj[0].start:obj[0].stop, obj[1].start: obj[1].stop] = 1
#             boxes.append([obj[1].start, obj[0].start, obj[1].stop, obj[0].stop])
#
#     output_img[np.where(mask == 255)] = 255
#
#     stamp_location = None
#     if len(boxes) == 0:
#         stamp_location = None
#     elif len(boxes) == 1:
#         stamp_location = boxes[0]
#     elif len(boxes) > 1:
#         y_min = 999999
#         pos = -1
#         for i, box in enumerate(boxes):
#             [x1, y1, x2, y2] = box
#             if y1 < y_min:
#                 y_min = y1
#                 pos = i
#         stamp_location = boxes[pos]
#
#     return output_img, stamp_location
#
#
# def visualize_hocr(text, image, key):
#     tags = {'word': 'ocrx_word',
#             'line': 'ocr_line',
#             'par': 'ocr_par',
#             'area': 'ocr_carea'}
#     colors = {'word': (0, 255, 0),
#               'line': (255, 0, 0),
#               'par': (0, 0, 255),
#               'area': (255, 255, 0)}
#     assert key in tags.keys(), f'Key must be one of {tags.keys()}'
#     tag = tags[key]
#     color = colors[key]
#     query = f".//*[@class='{tag}']"
#     tree = ElementTree.fromstring(text)
#     for elem in tree.findall(query):
#         bbox = elem.get('title').split(';')[0]
#         bbox = [int(val) for val in bbox.split()[1:]]
#         xmin, ymin, xmax, ymax = bbox
#         cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 3)
#     return image
#
#
# def visualize_cc(image, cc):
#     if len(image.shape) == 2:
#         image = np.dstack([image.copy()]*3)
#     image[np.where(image == 255)] = 127
#     minr, minc, maxr, maxc = cc.bbox
#     image[minr:maxr, minc:maxc] = (255, 0, 0)
#     image[list(zip(*cc.coords))] = (0, 255, 0)
#     return image
#
#
# def filter_regions_by_area(image):
#     min_threshold = MIN_AREA_RELATIVE_THRESHOLD * image.size
#     max_threshold = MAX_AREA_RELATIVE_THRESHOLD * image.size
#     regions = measure.label(image, background=255)
#     regions = measure.regionprops(regions, image)
#     for region in regions:
#         if region.area < min_threshold or region.area > max_threshold:
#             coords = list(zip(*region.coords))
#             image[coords] = 255
#
#
# def run_tesseract(image, lang='jpn_best', psm=11, hocr=True, tessdata_folder="/usr/share/tesseract-ocr/4.00/tessdata/"):
#     print("FLAX ----- Tesseract folder: ", tessdata_folder)
#     if psm == 2:
#         hocr = False
#     with tempfile.TemporaryDirectory() as temp_dir:
#         image_path = os.path.join(temp_dir, 'feelsgoodman.png')
#         imsave(image_path, image)
#         text = subprocess.run(
#                 'tesseract {} stdout -l {} --psm {} {} --tessdata-dir {}'.format(
#                     image_path, lang, psm, '-c tessedit_create_hocr=1' if hocr else '', tessdata_folder),
#                     shell=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
#         text = text.stdout.decode() if psm != 2 else text.stderr.decode()
#         text = normalize('NFKC', text)
#         if hocr:
#             print("FLAX ----- run_tesseract(): ", text)
#             return text
#         text = text.replace(' ', '')
#         text = text.replace('\n\n', '\n')
#         print("FLAX ----- run_tesseract(): ", text)
#         return text
#
#
# def normalize_orientation(image):
#     osd_result = run_tesseract(image, psm=0, tessdata_folder=TESSDATA_PATH)
#     degrees_of_orientation = int(osd_result.splitlines()[1].split(':')[-1])
#     if degrees_of_orientation != 0:
#         image = transform.rotate(image, degrees_of_orientation, resize=True)
#         image = (image * 255).astype('uint8')
#     return image
#
#
# def process_single_image(image_path, hocr=True):
#     image = imread(image_path, as_gray=True)
#     filter_regions_by_area(image)
#     image = normalize_orientation(image)
#     text = run_tesseract(image, hocr=hocr, tessdata_folder=TESSDATA_PATH)
#     return image, text
#
#
# def make_feature_dict(
#         text, corrected_text, xmin, ymin, xmax, ymax,
#         page_width, page_height, confidence_by_char, confidence_by_field):
#         return {
#                 'company_id':'',
#                 'line': text,
#                 'corrected_line': corrected_text,
#                 'xmin': xmin,
#                 'ymin': ymin,
#                 'xmax': xmax,
#                 'ymax': ymax,
#                 'page_width': page_width,
#                 'page_height': page_height,
#                 'confidence_by_char': confidence_by_char,
#                 'confidence_by_field': confidence_by_field
#                 }
#
#
# def find_best_match(text, candidate_list, mode='lms'):
#     '''Find best matching string using either longest matching sequence (lms)
#     or smallest edit distance (sed)'''
#     best_match_count = 0 if mode == 'lms' else 1000
#     best_candidate = ''
#     for candidate in candidate_list:
#         candidate_without_space = candidate.replace(' ', '')
#         if mode == 'lms':
#             match = difflib.SequenceMatcher(a=text, b=candidate_without_space)
#             match = match.find_longest_match(0, len(match.a), 0, len(match.b))
#             if match.size >= best_match_count:
#                 best_match_count = match.size
#                 best_candidate = candidate
#         elif mode == 'sed':
#             distance = editdistance.eval(text, candidate_without_space)
#             if distance < best_match_count:
#                 best_match_count = distance
#                 best_candidate = candidate
#         else:
#             raise Exception('Matching mode must be either "lms" or "sed"')
#     return best_candidate
#
#
# def match_pattern(pattern, doc):
#     '''Find occurences of `pattern` in given Document'''
#     for match in re.finditer(pattern, doc.text):
#         start = match.start()
#         end = match.end()
#         match_text = doc.text[start:end]
#         line_text = doc.get_line(start)
#         line_id = doc.get_line_id(start)
#         confidence_by_char = doc.confidence_by_char[line_id]
#         confidence_by_field = doc.confidence_by_field[line_id]
#         xmin, ymin, xmax, ymax = doc.idx2line[start]
#         feature_dict = make_feature_dict(
#                 line_text, match_text, xmin, ymin, xmax, ymax,
#                 doc.width, doc.height, confidence_by_char, confidence_by_field)
#         yield feature_dict
#
#
# def on_right_side(feature_dict, threshold=0.5):
#     '''Returns true if x is on the right side of the page'''
#     x = feature_dict['xmin']
#     page_width = feature_dict['page_width']
#     return True if x > threshold*page_width else False
#
#
# def tel_fax_post_process(text):
#     text = [char for char in text if char in '0123456789']
#     text.insert(2, '-')
#     text.insert(7, '-')
#     text = ''.join(text)
#     return text
#
#
# def detect_tel(doc):
#     '''Find phone number on a page'''
#     pattern = r'[0-9]{2}[\(\)\- ]+[0-9]{4}[\(\)\-一 ]+[0-9]{4}'
#     matches = list(match_pattern(pattern, doc))
#     matches = [match for match in matches if on_right_side(match)]
#     return matches[0:1] if matches else []
#
#
# def detect_fax(doc):
#     '''Find fax number on a page'''
#     pattern = r'[0-9]{2}[\(\)\- ]+[0-9]{4}[\(\)\-一 ]+[0-9]{4}'
#     matches = list(match_pattern(pattern, doc))
#     matches = [match for match in matches if on_right_side(match)]
#     return matches[1:2] if len(matches) >= 2 else []
#
#
# def identity(arg):
#     return arg
#
#
# def constant_true(*_):
#     return True
#
#
# def is_company(text):
#     if len(text) >= 35: # Pad length for classification model is set to 35
#         return False
#     predicted_label, _ = line_classification_model.predict(text)
#     return predicted_label == 'company'
#
#
# def is_on_right_half(xmin, page_width):
#     threshold = 0.5
#     return True if xmin/page_width > threshold else False
#
#
# def is_higher_than(threshold=0.5):
#     '''Check if value is higher than a threshold'''
#     return lambda value: value > threshold
#
#
# def detect_item(doc, filter_ops=[constant_true], reduce_ops=[identity]):
#     '''Find item on a page'''
#     candidates = list(line for line in doc)
#     for features, op in filter_ops:
#         candidates = [
#                 candidate for candidate in candidates if op(
#                     *[candidate[feature] for feature in features])]
#     for op in reduce_ops:
#         candidates = op(candidates)
#     return candidates
#
#
# def match_with_database(
#         lines, database, src_key='line', dst_key='corrected_line'):
#     '''Autocorrect using a database'''
#     for line in lines:
#         line[dst_key] = find_best_match(line[src_key], database)
#     return lines
#
#
# def match_with_company_database(lines):
#     global company_list
#     return match_with_database(lines, company_list)
#
#
# def match_with_account_database(lines):
#     global account_list
#     return match_with_database(lines, account_list)
#
#
# def find_company_id(lines, src_key='corrected_line', dst_key='company_id'):
#     global company_database
#     for line in lines:
#         line[dst_key] = company_database[line[src_key]]
#     return lines
#
#
# def find_account(lines):
#     global account_database
#     for line in lines:
#         line['corrected_line'] = account_database[line['corrected_line']]
#     return lines
#
#
# def detect_company(doc):
#     '''Detect issueing company name in a Japanese invoice document
#     Args:
#         doc: Document object
#     Returns:
#         result: feature dict of detected company name entity
#     '''
#     filter_ops = [
#             (('line',), is_company),
#             (('xmin', 'page_width'), is_on_right_half),
#             (('confidence_by_field',), is_higher_than(0.5))]
#     reduce_ops = [
#             take_longest,
#             match_with_company_database,
#             find_company_id]
#     result = detect_item(doc, filter_ops, reduce_ops)
#     return result
#
#
# def company_post_process(text):
#     return text.replace(' ', '')
#
#
# def is_address(text):
#     if len(text) >= 35: # Pad length for classification model is set to 35
#         return False
#     predicted_label, _ = line_classification_model.predict(text)
#     return predicted_label == 'address'
#
#
# def is_longer_than(threshold=3):
#     return lambda text: len(text) > threshold
#
#
# def is_vertically_overlapped(box1, box2):
#     _, ymin1, _, ymax1 = box1
#     _, ymin2, _, ymax2 = box2
#     return ymin1 < ymax2 and ymin2 < ymax1
#
#
# def is_horizontally_overlapped(box1, box2):
#     xmin1, _, xmax1, _ = box1
#     xmin2, _, xmax2, _ = box2
#     return xmin1 < xmax2 and xmin2 < xmax1
#
#
# def is_intersect(box1, box2):
#     '''Check if two boxes intersect'''
#     return is_vertically_overlapped(box1, box2) and is_horizontally_overlapped(box1, box2)
#
#
# def get_box(line):
#     return tuple(line[key] for key in ['xmin', 'ymin', 'xmax', 'ymax'])
#
#
# def expand_box(box, factor):
#     '''Expand box downward and rightward by given factor'''
#     assert factor >= 0, 'Must supply a positive factor'
#     xmin, ymin, xmax, ymax = box
#     width = xmax - xmin
#     height = ymax - ymin
#     new_xmax = int(xmin + width*factor)
#     new_ymax = int(ymin + height*factor)
#     return (xmin, ymin, new_xmax, new_ymax)
#
#
# def take_first_cluster(lines, factor=2.5, key=get_box):
#     '''Take the first lines that form a cluster
#     Args:
#         lines: list of dicts that expose box-like values
#         factor: factor to consider a line to be in the same cluster
#         key: function to get box values from each line
#     Returns:
#         lines that belong the the first cluster'''
#     result = lines[:1]
#     for i in range(1, len(lines)):
#         previous_line = result[-1]
#         current_line = lines[i]
#         previous_box = key(previous_line)
#         previous_box = expand_box(previous_box, factor)
#         current_box = key(current_line)
#         if is_intersect(previous_box, current_box):
#             result.append(current_line)
#         else:
#             break
#     return result
#
#
# def merge_boxes(box1, box2):
#     xmin1, ymin1, xmax1, ymax1 = box1
#     xmin2, ymin2, xmax2, ymax2 = box2
#     new_xmin = min(xmin1, xmin2)
#     new_ymin = min(ymin1, ymin2)
#     new_xmax = max(xmax1, xmax2)
#     new_ymax = max(ymax1, ymax2)
#     return new_xmin, new_ymin, new_xmax, new_ymax
#
#
# def merge_lines(lines):
#     '''Merge line entities into one'''
#     if not lines:
#         return []
#     text = reduce(add, map(itemgetter('line'), lines))
#     corrected_text = reduce(add, map(itemgetter('corrected_line'), lines))
#     xmin, ymin, xmax, ymax = reduce(merge_boxes, map(get_box, lines))
#     page_width = lines[0]['page_width']
#     page_height = lines[0]['page_height']
#     confidence_by_char = reduce(add, map(itemgetter('confidence_by_char'), lines))
#     confidence_by_field = np.mean(confidence_by_char)
#     return [make_feature_dict(
#         text, corrected_text, xmin, ymin, xmax, ymax,
#         page_width, page_height, confidence_by_char, confidence_by_field
#         )]
#
#
# def take_longest(lines):
#     '''Take the longest line in the list'''
#     return [max(lines, key=lambda x: len(x['line']))] if lines else []
#
#
# def detect_address(doc):
#     filter_ops = [
#         (('line',), is_address),
#         (('line',), is_longer_than(3)),
#         (('xmin', 'page_width'), is_on_right_half),
#         (('confidence_by_field',), is_higher_than(0.5))]
#     reduce_ops = [
#         take_first_cluster,
#         merge_lines]
#     result = detect_item(doc, filter_ops, reduce_ops)
#     return result
#
#
# def address_post_process(text):
#     return identity(text)
#
#
# def detect_account(doc):
#     filter_ops = [
#             (('line',), is_company),
#             (('xmin', 'page_width'), is_on_right_half),
#             (('confidence_by_field',), is_higher_than(0.5))]
#     reduce_ops = [
#             take_longest,
#             match_with_company_database,
#             find_account]
#     return detect_item(doc, filter_ops, reduce_ops)
#
#
# def account_post_process(text):
#     return text
#
#
# def double_lines(text):
#     '''Function to double each line in ocr result for easier manual labelling'''
#     return '\n'.join(line + '\n' + line for line in text.split('\n'))
#
#
# def format_company_id(entity):
#     return entity['company_id']
#
#
# def format_line_text(entity):
#     return entity['line']
#
#
# def format_corrected_line_text(entity):
#     return entity['corrected_line']
#
#
# def format_box(entity):
#     return ','.join(str(entity[key]) for key in ['xmin', 'ymin', 'xmax', 'ymax'])
#
#
# def format_confidence_by_char(entity):
#     return ','.join(f'{confidence:.2f}' for confidence in entity['confidence_by_char'])
#
#
# def format_confidence_by_field(entity):
#     return f'{entity["confidence_by_field"]:.2f}'
#
#
# def strip_fields(result_dict, whitelist):
#     '''Strip fields from result dict'''
#     return {key: value for key, value in result_dict.items() if key in whitelist}
#
#
# def format_result(entity):
#     company_id = format_company_id(entity)
#     result = format_line_text(entity)
#     corrected_result = format_corrected_line_text(entity)
#     box = format_box(entity)
#     confidence_by_char = format_confidence_by_char(entity)
#     confidence_by_field = format_confidence_by_field(entity)
#     return {"会社コード": f"{company_id}",
#             "自動補正結果": f"{corrected_result}",
#             "読取座標": f"{box}",
#             "読取文字別尤度": f"{confidence_by_char}",
#             "読取結果": f"{result}",
#             "読取項目別尤度": f"{confidence_by_field}"}
#
#
# def detect_all(image_path):
#     image_dir = os.path.dirname(image_path)
#     image_id = os.path.basename(image_path).split('.')[0]
#     result_path = os.path.join(image_dir, image_id + '.json')
#
#     result_dict = {key: [] for key in map(str, [1, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14])}
#
#     # Read input
#     image = read_pdf_image(image_path)
#     image, stamp_location = remove_red_color(image)
#     image = normalize_orientation(image)
#     image_path = os.path.join(image_dir, image_id + '_normed.png')
#     imsave(image_path, image)
#
#     # Run Anh's detection
#     global layout_model
#     intermediary_result, lines = detect_fields(image_path, layout_model, TESSDATA_PATH, BANK_BRANCH_MASTER_PATH)
#
#     for key, entities in intermediary_result.items():
#         for entity in entities:
#             if entity['読取結果'] != '':
#                 result_dict[key].append(entity)
#
#     # Run Nick's detection
#     doc = Document(lines)
#
#     detect_ops = [
#             detect_company,
#             detect_address,
#             detect_tel,
#             detect_fax,
#             detect_account]
#
#     post_process_ops = [
#             company_post_process,
#             address_post_process,
#             tel_fax_post_process,
#             tel_fax_post_process,
#             account_post_process]
#
#     field_keys = map(str, [
#             1,
#             12,
#             13,
#             14,
#             11])
#
#     for detect_op, post_process_op, field_key in zip(detect_ops, post_process_ops, field_keys):
#         for result in detect_op(doc):
#             result['corrected_line'] = post_process_op(result['corrected_line'])
#             result_dict[field_key].append(format_result(result))
#
#
#     try:
#         company_id = result_dict['1'][0]['会社コード']
#     except IndexError:
#         company_id = ''
#
#     for field, results in result_dict.items():
#         for result in results:
#             del result['会社コード']
#
#     # Keep only these fields as per CTC requirements
#     ctc_whitelist = list(map(str, [1, 3, 7, 8, 9, 10, 11]))
#     result_dict = strip_fields(result_dict, ctc_whitelist)
#
#
#     return result_dict
#
#
# def process_uploaded_image(task_id, image_path):
#     cut_cell_dir = os.path.join(settings.DEBUG_FILE_PATH, task_id)
#     data = {}
#     print("FLAX ------ Main Process - Processing id: {} , image path: {}".format(task_id, image_path))
#     image_dir = os.path.dirname(image_path)
#     image_id = os.path.basename(image_path).split('.')[0]
#     result_path = os.path.join(image_dir, image_id + '.json')
#
#     result_dict = {key: [] for key in map(str, [1, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14])}
#
#     # Read input
#     image = cv2.imread(image_path)
#     image, stamp_location = remove_red_color(image)
#     image = normalize_orientation(image)
#     image_path = os.path.join(image_dir, image_id + '_normed.png')
#     imsave(image_path, image)
#     print("FLAX ----- AFTER normalization()")
#     # Run Anh's detection
#     global layout_model
#     print("FLAX ----- Layout Model: ", layout_model)
#     intermediary_result, lines = detect_fields(image_path, layout_model, TESSDATA_PATH, BANK_BRANCH_MASTER_PATH)
#     print("FLAX ----- AFTER detect_fields(): intermediate result {}, lines {}".format(intermediary_result, lines))
#     for key, entities in intermediary_result.items():
#         for entity in entities:
#             if entity['読取結果'] != '':
#                 result_dict[key].append(entity)
#
#     # Run Nick's detection
#     doc = Document(lines)
#
#     detect_ops = [
#             detect_company,
#             detect_address,
#             detect_tel,
#             detect_fax,
#             detect_account]
#
#     post_process_ops = [
#             company_post_process,
#             address_post_process,
#             tel_fax_post_process,
#             tel_fax_post_process,
#             account_post_process]
#
#     field_keys = map(str, [
#             1,
#             12,
#             13,
#             14,
#             11])
#
#     for detect_op, post_process_op, field_key in zip(detect_ops, post_process_ops, field_keys):
#         for result in detect_op(doc):
#             result['corrected_line'] = post_process_op(result['corrected_line'])
#             result_dict[field_key].append(format_result(result))
#
#     print("FLAX ----- AFTER detect functions: result dict".format(str(result_dict)))
#     try:
#         company_id = result_dict['1'][0]['会社コード']
#     except IndexError:
#         company_id = ''
#
#     for field, results in result_dict.items():
#         for result in results:
#             del result['会社コード']
#
#     # Keep only these fields as per CTC requirements
#     ctc_whitelist = list(map(str, [1, 3, 7, 8, 9, 10, 11]))
#     result_dict = strip_fields(result_dict, ctc_whitelist)
#     print("FLAX ----- FINAL: result dict".format(str(result_dict)))
#     return result_dict
#
#
# if __name__ == '__main__':
#     IMAGE_DIR = 'data/converted'
#     image_paths = sorted(glob(IMAGE_DIR + '/*'))
#     for image_path in image_paths:
#         if os.path.basename(image_path).split('.')[0] != '00014':
#             continue
#         start_time = time.time()
#         result = detect_all(image_path)
#         stop_time = time.time()
#         pprint(result)
#         #json.dump(result, open(result_path, 'w'), ensure_ascii=False, indent=4, sort_keys=True)
#         print(f'done {image_path}, took {stop_time - start_time} secs')
#         break
