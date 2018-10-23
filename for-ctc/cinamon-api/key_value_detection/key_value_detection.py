import numpy as np
import random
import os, json, re
import cv2
import difflib, editdistance
import pandas as pd
import unicodedata
from key_value_detection import line_cut
import sys
from key_value_detection.layout_model.layout_model import LayoutModel
sys.path.extend(['../table_analysis', '../field_classification_model'])
from table_analysis.line_cut_expending import expend_line_cut
import time


LAYOUT_RESULT_FOLDER = 'result'
LINE_JSON_FILE_NAME = 'data.json'
FIX_LINE_JSON_FILE_NAME = 'data_new.json'
LINE_CUT_FOLDER = 'linecut'
OCR_RESULT_JSON_FILE = 'ocr_result.json'


def draw_image_of_linecut(lines, image_file, folder):
    """ Draw image of lines """
    image = cv2.imread(image_file)

    if not os.path.exists(folder):
        os.mkdir(folder)
    image_width = image.shape[1]
    image_height = image.shape[0]

    for k, v in lines.items():
        [y1, x1, y2, x2] = v['location']
        x1, y1 = x1-2, y1-2
        x2, y2 = x2+2, y2+2
        if y1 < 0: y1 = 0
        if x1 < 0: x1 = 0
        if x2 > image_width: x2 = image_width
        if y2 > image_height: y2 = image_height
        line_image = image[y1:y2, x1:x2]
        line_image_path = os.path.join(folder, k + '.png')
        cv2.imwrite(line_image_path, line_image)


def get_info_cells(text_lines, lines_location):
    """Get text and information of cells"""

    cells = {}
    group_line = {}
    for key, value in text_lines.items():
        if 'cell' in key:
            key_arr = key.split('_')
            cell_name = key_arr[0] + '_' + key_arr[1]
            if cell_name not in group_line:
                group_line[cell_name] = [{
                    'line':key,
                    'text':value['text'],
                    'confidences': value['confidences'],
                    'location':lines_location[key]['location']
                }]
            else:
                group_line[cell_name].append({
                    'line':key,
                    'text':value['text'],
                    'confidences': value['confidences'],
                    'location':lines_location[key]['location']
                })
    for key, value in group_line.items():
        # if cell contains 1 line
        if len(value) == 1:
            cells[key] = {
                'text': value[0]['text'],
                'confidences': value[0]['confidences'] ,
                'location': value[0]['location']
            }
        else:
            # sort lines in cell by y1, x1
            value.sort(key=lambda x: (x['location'][0], x['location'][1]))

            rows = []
            current_line = value[0]
            arr_line = [current_line]
            for i, line in enumerate(value):
                if i==0: continue
                y1,_,_,_ = line['location']
                _,_,y2_current,_ = current_line['location']
                # if 2 line intersect together
                if y2_current - y1 - 10 > 0:
                    arr_line.append(line)
                else:
                    rows.append(arr_line)
                    current_line = line
                    arr_line = [current_line]
            rows.append(arr_line)

            text = ''
            confidences = []
            x1,y1,x2,y2 = 0,0,0,0
            for i, row in enumerate(rows):
                row.sort(key=lambda x: x['location'][1])
                if i == 0:
                    y1, x1 = row[0]['location'][0], row[0]['location'][1]
                if i == len(rows) - 1:
                    y2, x2 = row[-1]['location'][2], row[-1]['location'][3]
                for line in row:
                    text += line['text']
                    confidences.extend(line['confidences'])
            cells[key] = {'text': text, 'confidences': confidences, 'location': [y1,x1,y2,x2]}
    return cells


def get_info_lines(text_lines, lines_location):
    """Get text and information of lines"""
    lines_text = {}
    for key, value in text_lines.items():
        if 'text_line' in key:
            location = lines_location[key]['location']
            lines_text[key] = {'text': value['text'], 'location': location, 'confidences': value['confidences']}
    return lines_text


def getMoney(cell):
    """Get money and confidence in text"""
    text = cell['text']
    confidences = cell['confidences']
    objs = re.finditer(r"\d{1,3}[ ,]*(?:[,\.] *\d{3} *)+", text)

    money = []
    for obj in objs:
        matched_text = text[obj.start(): obj.end()]
        confidence = confidences[obj.start(): obj.end()]
        money.append({
            'text': matched_text,
            'confidences': confidence
        })
    return money


def detect_money(data):
    """Detect sum, tax, total"""

    list_number = []
    for cell_name, cell in data.items():
        money = getMoney(cell)
        if len(money) > 0:
            for m in money:
                number_text = re.sub("[^0-9]", "", m['text'])
                number_value = float(number_text)
                number_confidences = m['confidences']
                string_confidences = []
                for conf in number_confidences:
                    string_confidences.append('{:.2f}'.format(conf))
                confidence_by_field = '{:.2f}'.format(sum(number_confidences) / len(number_confidences))
                confidence_by_char = ','.join(string_confidences)
                dict_value = {
                    'text': m['text'],
                    'confidence_by_char': confidence_by_char,
                    'confidence_by_field': confidence_by_field,
                    'number_text': number_text,
                    'number_value': number_value,
                    'location': cell['location']
                }
                list_number.append(dict_value)
    final = {}
    empty_dict = {
        'autocorrect': '',
        'box': None,
        'ocr': '',
        'confidence_by_char': '',
        'confidence_by_field': ''
    }
    if len(list_number) == 0:
        final['total'] = empty_dict
    else:
        # get total
        list_number.sort(key=lambda x: -x['number_value'])
        total_autocorrect = list_number[0]['number_text']
        total_box = list_number[0]['location']
        total_ocr = list_number[0]['text']
        total_conf_by_char = list_number[0]['confidence_by_char']
        total_conf_by_field = list_number[0]['confidence_by_field']
        final['total'] = {
            'autocorrect': total_autocorrect,
            'box': total_box,
            'ocr': total_ocr,
            'confidence_by_char': total_conf_by_char,
            'confidence_by_field': total_conf_by_field
        }
    return final


def detect_bank(key, value, list_bank_names, line_banks = []):
    """Detect bank by keyword"""
    list_banks = []
    lineid = 0
    if 'line' in key:
        lineid = int(key.split('line')[1])
    bank_objs = re.finditer(r"\w+銀行|ミ[ズス]ホ|りそな|三井住友|三菱東京UFJ", value['text'])
    for bank_obj in bank_objs:
        obj = value['text'][bank_obj.start():bank_obj.end()]
        confidences = value['confidences'][bank_obj.start():bank_obj.end()]

        if len(obj) < 10:
            info = {
                'bank_name': obj,
                'raw_bank_name':None,
                'confidences': confidences,
                'data': value,
                'pos': 0,
                'name': key,
                'lineid': lineid
            }
            list_banks.append(info)
        elif '銀行' in obj:
            start = 0
            indices = [m.end() for m in re.finditer('銀行', obj)]
            for index in indices:
                info = {
                    'bank_name': None,
                    'raw_bank_name': None,
                    'confidences': None,
                    'data': value,
                    'pos': index,
                    'name': key,
                    'lineid': lineid
                }
                temp_obj = obj[start:index]
                temp_confidences = confidences[start:index]
                start = index
                if len(temp_obj) < 10:
                    info['bank_name'] = temp_obj
                    info['confidences'] = temp_confidences
                else:
                    info['bank_name'] = temp_obj[-10:]
                    info['confidences'] = temp_confidences[-10:]
                list_banks.append(info)

    if len(list_banks) == 0:
        text = value['text'].replace('銀行','')
        best_matched_text, best_score = find_best_match(text, list_bank_names, mode='sed')
        best_matched_text_temp = best_matched_text
        best_score_temp = best_score
        if '銀行' in best_matched_text and '銀行' in value['text']:
            best_matched_text_temp = best_matched_text.replace('銀行','')
            best_score_temp = best_score - 2
        max_len = max(len(text), len(best_matched_text_temp))
        if best_score_temp / max_len <= 0.2:
            info = {
                'bank_name': best_matched_text,
                'raw_bank_name': value['text'],
                'confidences': value['confidences'],
                'data': value,
                'pos': 0,
                'name': key,
                'lineid': lineid
            }
            list_banks.append(info)
    if len(list_banks) > 0:
        line_banks.extend(list_banks)


def detect_branch(key, value, list_branch_names, line_branches = []):
    """Detect branch by keyword"""
    list_branches = []
    lineid = 0
    if 'line' in key:
        lineid = int(key.split('line')[1])
    branch_objs = re.finditer(r'\w+(?:支店)|\w+(?:営業部)|(?:新東京)', value['text'])
    for branch_obj in branch_objs:
        obj = value['text'][branch_obj.start():branch_obj.end()]
        confidences = value['confidences'][branch_obj.start():branch_obj.end()]
        indices = [m.end() for m in re.finditer('(?:支店)|(?:営業部)|(?:新東京)', obj)]
        start = 0
        for index in indices:
            info = {
                'branch_name': None,
                'raw_branch_name': None,
                'confidences': None,
                'data': value,
                'pos': index,
                'name': key,
                'lineid': lineid
            }
            temp_obj = obj[start:index]
            temp_confidences = confidences[start:index]
            start = index
            # find index.start of words in string, to get new string
            idx_temp_obj = [m.end() for m in
                            re.finditer("(?:銀行)|(?:りそな)|(?:三井住友)|(?:ミ[ズス]ホ)|(?:三菱東京UFJ)", temp_obj)]
            if len(idx_temp_obj) > 0:
                idx = max(idx_temp_obj)
                new_temp_obj = temp_obj[idx:]
                new_temp_confidences = temp_confidences[idx:]
            else:
                if len(temp_obj) > 10:
                    new_temp_obj = temp_obj[-10:]
                    new_temp_confidences = temp_confidences[-10:]
                else:
                    new_temp_obj = temp_obj
                    new_temp_confidences = temp_confidences
            info['branch_name'] = new_temp_obj
            info['confidences'] = new_temp_confidences
            list_branches.append(info)
    if len(list_branches) == 0:
        text = value['text'].replace('支店', '')
        best_matched_text, best_score = find_best_match(text, list_branch_names, mode='sed')
        best_matched_text_temp = best_matched_text
        best_score_temp = best_score
        if '支店' in best_matched_text and '支店' in value['text']:
            best_matched_text_temp = best_matched_text.replace('支店', '')
            best_score_temp = best_score - 2
        max_len = max(len(text), len(best_matched_text_temp))
        if best_score_temp / max_len <= 0.2:
            info = {
                'branch_name': best_matched_text,
                'raw_branch_name':value['text'],
                'confidences': value['confidences'],
                'data': value,
                'pos': 0,
                'name': key,
                'lineid': lineid
            }
            list_branches.append(info)

    if len(list_branches) > 0:
        line_branches.extend(list_branches)


def detect_account_type_number(key, value, line_type_numbers = []):
    """Detect account_type, account_number by keyword"""
    type_number_list = []
    lineid = 0
    if 'line' in key:
        lineid = int(key.split('line')[1])
    type_numer_objs = re.finditer(r'(?:普通預金|通預金|普通口座|普通|普|当座預金|当座|当).{0,10}(?:\**\d{3,9}\**)', value['text'])
    acc_type = ''
    acc_type_confidences = None
    acc_num = ''
    acc_num_confidences = None
    index = 0
    for type_numer_obj in type_numer_objs:
        obj = value['text'][type_numer_obj.start():type_numer_obj.end()]
        confidences = value['confidences'][type_numer_obj.start():type_numer_obj.end()]
        type_obj = re.search(r'(?:普通預金|通預金|普通口座|普通|普|当座預金|当座|当\w|当)', obj)
        if type_obj:
            acc_type = type_obj.group()
            acc_type_confidences = confidences[type_obj.start():type_obj.end()]
            if acc_type == '通預金':
                acc_type = '普' + acc_type
                acc_type_confidences = [1.0] + acc_type_confidences
            if len(acc_type) == 2 and '当' in acc_type:
                acc_type = '当座'
        num_obj = re.search(r'(?:\**\d{3,9}\**)', obj)
        if num_obj:
            acc_num = num_obj.group()
            acc_num_confidences = confidences[num_obj.start():num_obj.end()]

        info = {
            'account_type': acc_type,
            'account_type_confidences': acc_type_confidences,
            'account_number': acc_num,
            'account_number_confidences': acc_num_confidences,
            'data': value,
            'pos': index,
            'name': key,
            'lineid': lineid
        }
        type_number_list.append(info)
        index += 1
    if len(type_number_list) == 0:
        objs = re.finditer(r'(?:支店).{1,6}\d{5,9}|(?:営業部).{1,6}\d{5,9}|(?:新東京).{1,6}\d{5,9}', value['text'])
        for obj in objs:
            text = value['text'][obj.start():obj.end()]
            text_confidences = value['confidences'][obj.start():obj.end()]
            prefix_obj = re.search("(?:支店)|(?:営業部)|(?:新東京)", text)
            if prefix_obj:
                num_type_text = text[prefix_obj.end():]
                num_type_confidences = text_confidences[prefix_obj.end():]
                num_obj = re.search("\d{5,9}", num_type_text)
                if num_obj:
                    type_text = num_type_text[:num_obj.start()]
                    tye_confidences = num_type_confidences[:num_obj.start()]
                    num_text = num_type_text[num_obj.start():]
                    num_confidences = num_type_confidences[num_obj.start():]

                    info = {
                        'account_type': type_text,
                        'account_type_confidences': tye_confidences,
                        'account_number': num_text,
                        'account_number_confidences': num_confidences,
                        'data': value,
                        'pos': index,
                        'name': key,
                        'lineid': lineid
                    }
                    type_number_list.append(info)
                    index += 1
    if len(type_number_list) == 0:
        if value['text'].isdigit() and len(value['text'])==7:
            info = {
                'account_type': None,
                'account_type_confidences': None,
                'account_number': value['text'],
                'account_number_confidences': value['confidences'],
                'data': value,
                'pos': 0,
                'name': key,
                'lineid': lineid
            }
            type_number_list.append(info)
    if len(type_number_list) > 0:
        line_type_numbers.extend(type_number_list)


def find_best_match(text, candidate_list, mode='lms'):
    '''Find best matching string using either longest matching sequence (lms)
    or smallest edit distance (sed)'''
    best_match_count = 0 if mode == 'lms' else 1000
    best_candidate = ''
    for candidate in candidate_list:
        candidate_without_space = candidate.replace(' ', '')
        if mode == 'lms':
            match = difflib.SequenceMatcher(a=text, b=candidate_without_space)
            match = match.find_longest_match(0, len(match.a), 0, len(match.b))
            if match.size > best_match_count:
                best_match_count = match.size
                best_candidate = candidate
        else:
            distance = editdistance.eval(text, candidate_without_space)
            if distance < best_match_count:
                best_match_count = distance
                best_candidate = candidate
    return best_candidate, best_match_count


def make_text_to_code_dict(dataframe, text_columns, code_column):
    '''Make mapping from text to code'''
    result_dict = {}
    code_column = dataframe.loc[:,code_column].fillna('')
    code_column = list(code_column)
    code_column = [unicodedata.normalize('NFKC', str(code)) for code in code_column]
    for column in text_columns:
        text_column = dataframe.loc[:,column].fillna('')
        text_column = list(text_column)
        text_column = [unicodedata.normalize('NFKC', text) for text in text_column]
        for text, code in zip(text_column, code_column):
            result_dict[text] = code
    result_dict.pop('', None)
    return result_dict


def autocorrect_bank(line_banks, list_bank_names):
    """autocorrect bank by master data"""

    corrected_line_banks = []
    for i, line_bank in enumerate(line_banks):
        if line_bank['raw_bank_name']:
            info = {
                'bank_name': line_bank['bank_name'],
                'bank_name_raw': line_bank['raw_bank_name'],
                'confidences': line_bank['confidences'],
                'data': line_bank['data'],
                'pos': line_bank['pos'],
                'name': line_bank['name']}
            corrected_line_banks.append(info)
        else:
            bank_name = line_bank['bank_name'].replace('銀行','')
            best_matched_text, best_score = find_best_match(bank_name, list_bank_names, mode='sed')
            max_len = max(len(bank_name), len(best_matched_text))
            if best_score/max_len <= 0.8:
                data = line_bank['data']
                pos = line_bank['pos']
                name = line_bank['name']
                info = {
                    'bank_name': None,
                    'bank_name_raw': line_bank['bank_name'],
                    'confidences': line_bank['confidences'],
                    'data': data,
                    'pos': pos,
                    'name': name}

                if '三菱UFJ' in best_matched_text: best_matched_text = best_matched_text.replace('三菱UFJ', '三菱東京UFJ')
                if '銀行' in line_bank['bank_name'] and '銀行' not in best_matched_text:
                    bank_name = best_matched_text + '銀行'
                    info['bank_name'] = bank_name
                elif '銀行' not in line_bank['bank_name'] and '銀行' in best_matched_text:
                    info['bank_name'] = re.sub("銀行", "", best_matched_text)
                else:
                    info['bank_name'] = best_matched_text
                corrected_line_banks.append(info)
    return corrected_line_banks


def autocorrect_branch(line_branches, list_branch_names):
    """autocorrect branch by master data"""

    corrected_line_branches = []
    for i, line_branch in enumerate(line_branches):
        if line_branch['raw_branch_name']:
            info = {
                'branch_name': line_branch['branch_name'],
                'branch_name_raw': line_branch['raw_branch_name'],
                'confidences': line_branch['confidences'],
                'data': line_branch['data'],
                'pos': line_branch['pos'],
                'name': line_branch['name']}
        else:
            branch_name = re.sub("(?:支店)|(?:営業部)|\d+", "", line_branch['branch_name'])
            best_matched_text, best_score = find_best_match(branch_name, list_branch_names, mode='sed')
            max_len = max(len(branch_name), len(best_matched_text))
            if best_score / max_len <= 0.8:
                data = line_branch['data']
                pos = line_branch['pos']
                name = line_branch['name']
                info = {
                    'branch_name': None,
                    'branch_name_raw': line_branch['branch_name'],
                    'confidences': line_branch['confidences'],
                    'data': data,
                    'pos': pos,
                    'name': name}

                if '支店' in line_branch['branch_name'] and '支店' not in best_matched_text:
                    branch_name = best_matched_text + '支店'
                    info['branch_name'] = branch_name
                elif '営業部' in line_branch['branch_name'] and '営業部' not in best_matched_text:
                    branch_name = best_matched_text + '営業部'
                    info['branch_name'] = branch_name
                elif ('支店' not in line_branch['branch_name'] and '支店' in best_matched_text) or \
                    ('営業部' not in line_branch['branch_name'] and '営業部' in best_matched_text):
                    info['branch_name'] = re.sub("(?:支店)|(?:営業部)", "", best_matched_text)
                else:
                    info['branch_name'] = best_matched_text
                corrected_line_branches.append(info)
    return corrected_line_branches


def convert_to_same_format(banks, branches, type_numbers):
    """Convert to same format json"""

    data_dict = {}
    count_bank = 1
    for bank in banks:
        key = 'bank_' + str(count_bank)
        count_bank += 1
        string_confidences = []
        for conf in bank['confidences']:
            string_confidences.append('{:.2f}'.format(conf))
        confidence_by_field = '{:.2f}'.format(sum(bank['confidences']) / len(bank['confidences']))
        confidence_by_char = ','.join(string_confidences)
        data_dict[key] = {
            'autocorrect': bank['bank_name'],
            'box': bank['data']['location'],
            'ocr': bank['bank_name_raw'],
            'confidence_by_char': confidence_by_char,
            'confidence_by_field': confidence_by_field
        }

    count_branch = 1
    for branch in branches:
        key = 'branch_' + str(count_branch)
        count_branch += 1

        string_confidences = []
        for conf in branch['confidences']:
            string_confidences.append('{:.2f}'.format(conf))
        confidence_by_field = '{:.2f}'.format(sum(branch['confidences']) / len(branch['confidences']))
        confidence_by_char = ','.join(string_confidences)
        data_dict[key] = {
            'autocorrect': branch['branch_name'],
            'box': branch['data']['location'],
            'ocr': branch['branch_name_raw'],
            'confidence_by_char': confidence_by_char,
            'confidence_by_field': confidence_by_field
        }

    count_type = 1
    for type_number in type_numbers:
        if type_number['account_type']:
            key_type = 'acc_type_' + str(count_type)
            type_string_confidences = []
            for conf in type_number['account_type_confidences']:
                type_string_confidences.append('{:.2f}'.format(conf))
            type_confidence_by_field = '{:.2f}'.format(sum(type_number['account_type_confidences']) /
                                                       len(type_number['account_type_confidences']))
            type_confidence_by_char = ','.join(type_string_confidences)
            data_dict[key_type] = {
                'autocorrect': type_number['account_type'],
                'box': type_number['data']['location'],
                'ocr': type_number['account_type'],
                'confidence_by_char': type_confidence_by_char,
                'confidence_by_field': type_confidence_by_field
            }
            count_type += 1

    count_number = 1
    for type_number in type_numbers:
        if type_number['account_number']:
            key_num = 'account_num_' + str(count_number)
            num_string_confidences = []
            for conf in type_number['account_number_confidences']:
                num_string_confidences.append('{:.2f}'.format(conf))
            num_confidence_by_field = '{:.2f}'.format(sum(type_number['account_number_confidences']) /
                                                       len(type_number['account_number_confidences']))
            num_confidence_by_char = ','.join(num_string_confidences)
            data_dict[key_num] = {
                'autocorrect': type_number['account_number'],
                'box': type_number['data']['location'],
                'ocr': type_number['account_number'],
                'confidence_by_char': num_confidence_by_char,
                'confidence_by_field': num_confidence_by_field
            }
            count_number += 1
    return data_dict

def get_account_type(account_type):
    if any(char in '普通' for char in account_type):
        return '1'
    elif any(char in '当座' for char in account_type):
        return '2'
    else:
        return '1'

def convert_to_json_result(data, bank_names_to_code_dict, branch_names_to_code_dict):
    """Convert to output format"""
    result = {}
    for key, value in data.items():
        autocorrect = value['autocorrect']
        ocr = value['ocr']
        confidence_by_char = value['confidence_by_char']
        confidence_by_field = value['confidence_by_field']
        box = ''
        if value['box']:
            box = '{0},{1},{2},{3}'.format(value['box'][1], value['box'][0], value['box'][3], value['box'][2])

        if key == 'total':
            result['3'] = [{
                '会社コード': '',
                '自動補正結果': autocorrect,
                '読取座標': box,
                '読取文字別尤度': confidence_by_char,
                '読取結果': ocr,
                '読取項目別尤度': confidence_by_field
            }]
        elif key == 'sum':
            result['4'] = [{
                '会社コード': '',
                '自動補正結果': autocorrect,
                '読取座標': box,
                '読取文字別尤度': confidence_by_char,
                '読取結果': ocr,
                '読取項目別尤度': confidence_by_field
            }]
        elif key == 'tax':
            result['5'] = [{
                '会社コード': '',
                '自動補正結果': autocorrect,
                '読取座標': box,
                '読取文字別尤度': confidence_by_char,
                '読取結果': ocr,
                '読取項目別尤度': confidence_by_field
            }]
        elif 'bank' in key:
            if '7' in result:
                result['7'].append({
                    '会社コード': '',
                    '自動補正結果': autocorrect,
                    '読取座標': box,
                    '読取文字別尤度': confidence_by_char,
                    '読取結果': ocr,
                    '読取項目別尤度': confidence_by_field
                })
            else:
                result['7'] = [{
                    '会社コード': '',
                    '自動補正結果': autocorrect,
                    '読取座標': box,
                    '読取文字別尤度': confidence_by_char,
                    '読取結果': ocr,
                    '読取項目別尤度': confidence_by_field
                }]
        elif 'branch' in key:
            if '8' in result:
                result['8'].append({
                    '会社コード': '',
                    '自動補正結果': autocorrect,
                    '読取座標': box,
                    '読取文字別尤度': confidence_by_char,
                    '読取結果': ocr,
                    '読取項目別尤度': confidence_by_field
                })
            else:
                result['8'] = [{
                    '会社コード': '',
                    '自動補正結果': autocorrect,
                    '読取座標': box,
                    '読取文字別尤度': confidence_by_char,
                    '読取結果': ocr,
                    '読取項目別尤度': confidence_by_field
                }]
        elif 'acc_type' in key:
            autocorrect = get_account_type(autocorrect)
            if '9' in result:
                result['9'].append({
                    '会社コード': '',
                    '自動補正結果': autocorrect,
                    '読取座標': box,
                    '読取文字別尤度': confidence_by_char,
                    '読取結果': ocr,
                    '読取項目別尤度': confidence_by_field
                })
            else:
                result['9'] = [{
                    '会社コード': '',
                    '自動補正結果': autocorrect,
                    '読取座標': box,
                    '読取文字別尤度': confidence_by_char,
                    '読取結果': ocr,
                    '読取項目別尤度': confidence_by_field
                }]
        elif 'account_num' in key:
            if '10' in result:
                result['10'].append({
                    '会社コード': '',
                    '自動補正結果': autocorrect,
                    '読取座標': box,
                    '読取文字別尤度': confidence_by_char,
                    '読取結果': ocr,
                    '読取項目別尤度': confidence_by_field
                })
            else:
                result['10'] = [{
                    '会社コード': '',
                    '自動補正結果': autocorrect,
                    '読取座標': box,
                    '読取文字別尤度': confidence_by_char,
                    '読取結果': ocr,
                    '読取項目別尤度': confidence_by_field
                }]
    return result


def get_line_information(text_lines, lines_location, page_width, page_height):
    """Get information of lines"""
    lines = []
    for key, value in text_lines.items():
        text = value['text']
        confidence_by_char = value['confidences']
        confidence_by_field = None
        box = ()
        if key in lines_location:
            location = lines_location[key]['location']
            box = (location[1], location[0], location[3], location[2])
        if len(confidence_by_char) > 0:
            confidence_by_field = sum(confidence_by_char)/len(confidence_by_char)
        lines.append({
            'line': text,
            'box': box,
            'page_width': page_width,
            'page_height': page_height,
            'confidence_by_char': confidence_by_char,
            'confidence_by_field': confidence_by_field
        })
    return lines


def detect_fields(image_file, layout_model, tessdata_folder, bank_branch_master_path):
    root_folder = os.path.dirname(image_file)
    file_basename = os.path.basename(image_file).split('.')[0]

    # get list of bank name
    bank_branch_master_df = pd.read_csv(bank_branch_master_path)
    bank_names_to_code_dict = make_text_to_code_dict(bank_branch_master_df, ['bank name', 'bank name_kana'],
                                                     'bank code')
    list_bank_names = list(bank_names_to_code_dict.keys())

    # get list of branch name
    branch_names_to_code_dict = make_text_to_code_dict(bank_branch_master_df, ['brunch name', 'brunch name_kana'],
                                                       'brunch code')
    list_branch_names = list(branch_names_to_code_dict.keys())
    # Do linecut
    line_cut.process_one_image(image_file, layout_model)
    # fix line
    data_json_file = os.path.join(root_folder, LAYOUT_RESULT_FOLDER + '_' + file_basename, LINE_JSON_FILE_NAME)
    new_data_json_file = os.path.join(root_folder, LAYOUT_RESULT_FOLDER + '_' + file_basename, FIX_LINE_JSON_FILE_NAME)
    _,(page_width, page_height) = expend_line_cut(data_json_file, new_data_json_file, image_file, debug=False)

    with open(new_data_json_file) as f:
        lines_location = json.load(f)

    # run tesseract ocr
    linecut_folder = os.path.join(root_folder, LINE_CUT_FOLDER + '_' + file_basename)
    draw_image_of_linecut(lines_location, image_file, linecut_folder)
    text_lines = line_cut.run_hocr(linecut_folder, model_folder=tessdata_folder)
    start = time.time()
    line_informations = get_line_information(text_lines, lines_location, page_width, page_height)

    # save ocr result
    ocr_json_file = os.path.join(linecut_folder, OCR_RESULT_JSON_FILE)
    with open(ocr_json_file, 'w') as f:
        json.dump(text_lines, f)

    # get text and info of cell and line
    cells_text = get_info_cells(text_lines, lines_location)
    lines_text = get_info_lines(text_lines, lines_location)
    data = {**cells_text, **lines_text}

    # detect sum, tax, total
    dict_money = detect_money(data)

    # detect bank, branch, account type, account number
    line_banks = []
    line_branches = []
    line_type_numbers = []
    for key, value in data.items():
        detect_bank(key, value, list_bank_names,line_banks)
        detect_branch(key, value, list_branch_names, line_branches)
        detect_account_type_number(key, value, line_type_numbers)

    # sort by lineid and postion
    line_banks.sort(key=lambda x: (x['lineid'], x['pos']))
    line_branches.sort(key=lambda x: (x['lineid'], x['pos']))
    line_type_numbers.sort(key=lambda x: (x['lineid'], x['pos']))

    # autocorrect bank, branch
    corrected_line_banks = autocorrect_bank(line_banks, list_bank_names)
    corrected_line_branches = autocorrect_branch(line_branches, list_branch_names)

    # convert to same format with money fields
    dict_data = convert_to_same_format(corrected_line_banks, corrected_line_branches, line_type_numbers)

    data_final = {**dict_money, **dict_data}
    result = convert_to_json_result(data_final, bank_names_to_code_dict, branch_names_to_code_dict)
    stop = time.time()
    print('Predict time (sum, bank, branch, account type, account number): {}'.format(stop - start))
    return result, line_informations


def main():
    image_file = '/Users/anh/Downloads/ctc/denoise/請求書_12/denoise.png'
    textline_model_path = '/Users/anh/Downloads/test_bprost/model/ff_model_textline.pb'
    border_model_path = '/Users/anh/Downloads/test_bprost/model/ff_model_border.pb'
    tessdata_folder = '/Users/anh/tesseract/tessdata'
    bank_branch_master_path = '/Users/anh/Downloads/bank_branch_master.csv'

    layout_model = LayoutModel(
        path_to_pb=textline_model_path,
        scale=0.35,
        mode='L',
        list_path_to_pb_alt=[border_model_path]
    )

    result, line_informations  = detect_fields(image_file, layout_model, tessdata_folder, bank_branch_master_path)
    print(result)
    print(line_informations)

if __name__ == '__main__':
    main()
