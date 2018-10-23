# coding=utf-8
import os
import math
import re
import codecs
import json
import named_entity_classification as tc
import constant
import ngram_models
import copy
import regex_named_entities as rne
import cv2
import numpy as np
#import xls_report_key
import argparse
from collections import OrderedDict
import pdb
import unicodecsv as csv
import confusion_matrix as cm
output_json = {}
key_value = {}
MAX_LINE_X = 24  # do day toi da cua 1 line ngang
MAX_LINE_Y = 20  # do day toi da cua 1 line doc
keys_using_only_location = [u"車名",u"氏名",u"住所",u"生年月日"]




def normalize_output_text(text):
    output = []
    # text = re.sub("\f", "", text)
    if output.count(u"\n") > 3:
        text = re.sub(u"\t", "", text)
        text = re.sub(u"\r", "", text)
        text = re.sub(u"\v", "", text)
        text = re.sub(u" ", "", text)
    else:
        text = re.sub("\s+", "", text)
    text = re.sub(u"（", u"(", text)
    text = re.sub(u"）", u"(", text)
    text = re.sub(u"・", u".", text)
    text = re.sub(u"、", u",", text)
    text = re.sub(u".あり0なし", u".あり", text)
    text = re.sub(u"0あり.なし", u".なし", text)
    '''
    Text correction
    '''
    text = re.sub(u"烹制限", u"無制限", text)
    for i, c in enumerate(text):
        if ord(c) == ord(u'\u3099'):  # dakuten mark
            if i > 0: output[-1] = chr(ord(text[i - 1]) + 1)
        elif ord(c) == ord(u'\u309A'):  # handakuten mark
            if i > 0: output[-1] = chr(ord(text[i - 1]) + 2)
        elif 9311 <= ord(c) and ord(c) < 9321:  # 0-9
            output.append(chr(ord(c) + 48 - 9311))
        elif 9321 < ord(c) and ord(c) <= 9331:  # 10-19
            output.append("1")
            output.append(chr(ord(c) + 48 - 9321))
        else:
            output.append(c)
    return ''.join(output)


def distance(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

#input check a cell is right 
def is_right_side(a, b, threshold=100):
    x = (b[0] + b[2] - a[0] - a[2]) / 2
    l1 = distance((a[2], a[1]), (b[0], b[1]))
    l2 = distance((a[2], a[3]), (b[0], b[3]))
    if x > 0 and l1 + l2 < threshold:
        return True
    return False


def location_score(a, b):
    x = (b[1] + b[3] - a[1] - a[3]) / 2
    y = (b[0] + b[2] - a[0] - a[2]) / 2
    ans = 400.0 / math.sqrt(x * x + y * y + 150)
    if x < -400:
        ans -= 2.0
    if y < -400:
        ans -= 2.0
    l1 = distance((a[0], a[3]), (b[0], b[1]))
    l2 = distance((a[2], a[3]), (b[2], b[1]))
    if x > 0 and l1 + l2 < 100:
        ans += 8.0
    elif a[0] <= b[0] and b[2] <= a[2]:
        ans += 5.0
    l1 = distance((a[2], a[1]), (b[0], b[1]))
    l2 = distance((a[2], a[3]), (b[0], b[3]))
    if y > 0 and l1 + l2 < 100:
        ans += 4.0
    return ans

# find..
def key_type_distance(type_pattern, type_found):
    if re.search("\|", type_pattern):
        return max([key_type_distance(l, type_found) for l in type_pattern.split("|")])
    if type_pattern == type_found:
        return 1
    return 0
# Choose format of return value_list,debug_list from search_value
# debug = 0 , return only value
# debug = 1 , return both id and value
def update_value_list(cell,value_list,debug_list):
    global output_json
    output_json[cell]["is_used"] = True
    out_key = cell
    out_value = output_json[out_key]["value"]
    value_list.append(out_value)
    debug_list.append({out_key:out_value})
    return value_list,debug_list
# Return Value of a cell by value | location 
# input cell 
def search_value(cell, values, only_location, debug = 0):
    global output_json
    location = cell['location']
    ans = []
    value_list = []
    debug_list = []
    #Approach 1 : Use only location in search value
    if only_location:
        #print ("DEBUG: Running only location ")
        #print cell
        if len(cell["adjacent"]['right']) == 1:
            next_cell = cell["adjacent"]['right'][0]
            # if there is no down cell
            if( len(cell["adjacent"]['down']) == 0):
                value_list,debug_list = update_value_list(next_cell,value_list,debug_list)
                return value_list,debug_list
            else:
                bellow_cell = cell["adjacent"]['down'][0]
            #  Check if right cell is not used and below cell is used
            if not output_json[next_cell]["is_used"] and output_json[bellow_cell]["is_used"]:
                value_list,debug_list = update_value_list(next_cell,value_list,debug_list)
                return value_list,debug_list
            #  Check if right cell is used and below cell isnot used
            if output_json[next_cell]["is_used"] and not output_json[bellow_cell]["is_used"]:
                value_list,debug_list = update_value_list(bellow_cell,value_list,debug_list)
                return value_list,debug_list
            value_list,debug_list = update_value_list(next_cell,value_list,debug_list)
            return value_list,debug_list
        else:# default get the first right cell
            next_cell = cell["adjacent"]['right'][0]
            value_list,debug_list = update_value_list(next_cell,value_list,debug_list)
            return value_list,debug_list

    # If we have only one key lue
    if len(values) == 1:
        value = values[0]

        print ("DEBUG: Running HERE ")
        for next_cell in cell["adjacent"]['right']:
            cells = [next_cell]
            c = output_json[next_cell]
            while len(c["adjacent"]['right']) == 1:
                cells.append(c["adjacent"]['right'][0])
                c = output_json[c["adjacent"]['right'][0]]
            is_okay = True
            for c in cells:
                label, s = tc.value_classification(output_json[c]['value'])
                if label not in value.split("|") and not re.search("^(|-|。)$", output_json[c]['value']):
                    if not (value == "MONEY" and re.search(rne.no_format, output_json[c]['value'])):
                        is_okay = False
            if is_okay:
                for cell in cells:
                    value_list,debug_list = update_value_list(cell,value_list,debug_list)
                print("INFO : Detected {} value cells".format(len(value_list)))
                return value_list,debug_list
    # if we have multiple value
    for value in values:
        best_score = 0
        best_cell = ""
        for icell in sorted(output_json.keys()):
            if output_json[icell]["is_used"]:
                continue
            ll, s = tc.value_classification(output_json[icell]["value"])
            score = s * location_score(location, output_json[icell]["location"]) * key_type_distance(
                re.sub("\d+", "", value), ll)
            if score > best_score:
                best_score = score
                best_cell = icell
        if best_score > 0.3:
            value_list,debug_list = update_value_list(best_cell,value_list,debug_list)
        else:
            # Should implement extract complex value soon!
            # Need to implement accept Null value
            if re.search("(ADDRESS|NAME)", values[0]):
                if len(cell["adjacent"]['right']) == 1:
                    next_cell = cell["adjacent"]['right'][0]
                    if not output_json[next_cell]["is_used"]:
                        value_list,debug_list = update_value_list(next_cell,value_list,debug_list)
                        return value_list,debug_list
            if re.search("MONEY", values[0]):
                if len(cell["adjacent"]['right']) == 1:
                    next_cell = cell["adjacent"]['right'][0]
                    if not output_json[next_cell]["is_used"] and re.search(rne.money_format,
                                                                           output_json[next_cell]["value"]):
                        value_list,debug_list = update_value_list(next_cell,value_list,debug_list)
                        return value_list,debug_list
            if re.search("NONFLEET_GRADE", values[0]):
                if len(cell["adjacent"]['right']) == 1:
                    next_cell = cell["adjacent"]['right'][0]
                    if not output_json[next_cell]["is_used"] and re.search(rne.nonfleet_grade_format,
                                                                           output_json[next_cell]["value"]):
                        value_list,debug_list = update_value_list(next_cell,value_list,debug_list)
                        return value_list,debug_list
                if len(cell["adjacent"]['down']) == 1:
                    next_cell = cell["adjacent"]['down'][0]
                    if not output_json[next_cell]["is_used"] and re.search(rne.nonfleet_grade_format,
                                                                           output_json[next_cell]["value"]):
                        value_list,debug_list = update_value_list(next_cell,value_list,debug_list)
                        return value_list,debug_list
            if re.search("LICENSE_COLOR", values[0]):
                if len(cell["adjacent"]['right']) == 1:
                    next_cell = cell["adjacent"]['right'][0]
                    if not output_json[next_cell]["is_used"] and re.search(rne.license_color,
                                                                           output_json[next_cell]["value"]):
                        value_list,debug_list = update_value_list(next_cell,value_list,debug_list)
                        return value_list,debug_list

            ans.append("Cannot detect!")
            value_list = ['Cannot detect']
            debug_list = [{'no_id':'Cannot detect!'}]
    print("INFO : End - Detected {} value cells".format(len(value_list) ))
    return value_list,debug_list
#    return ans


def levenshtein(s1, s2):
    if len(s1) < len(s2):
        return levenshtein(s2, s1)

    # len(s1) >= len(s2)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def key_compare_score(key, sample):
    key = key.decode("utf-8")
    str_dis = levenshtein(key, sample)
    if len(key) > 10:
        return 1 - str_dis * 0.1
    if len(key) > 7:
        return 1 - str_dis * 0.2
    if len(key) > 4:
        return 1 - str_dis * 0.3
    return 1 - str_dis


def read_file_key_value(fname):
    input_file = file(fname, "r")
    key_value = json.loads(input_file.read().decode("utf-8"))
    return key_value


def get_json_from_file(path):
    global output_json
    output_json = {}
    with open(path + os.path.sep + "result.txt") as f:
        for line in f:  # read rest of lines
            t = [int(x) for x in line.split()]
            if t[1] == 0:
                continue
            fname = "table%d_cell%d" % (t[0], t[1])
            output_json[fname] = {"location": t[2:]}
            tmp_file = codecs.open(os.path.join(path, fname + ".txt"), "r", encoding='utf-8')
            text = tmp_file.read().replace('\n', '').replace('\f', '').replace('\r', '')
            text = normalize_output_text(text)
            output_json[fname]["value"] = text
            output_json[fname]["is_used"] = False
    return output_json


def print_table_cell(key_value_dict):
    for key in sorted(key_value_dict.keys()):
        print key, key_value_dict[key]['value']
    return


def print_result(result_dict):
    for key in result_dict.keys():
        print key, '----------- detected {} groups -------------'.format(len(result_dict[key]))
        for i in range(len(result_dict[key])):
            print unicode(result_dict[key][i]['value_key']), "-", result_dict[key][i]['id_key'], ":",
            for x in result_dict[key][i]['value']:
                if type(x) is dict:
                    id_cell = x.keys()[0]
                    value_cell = x.values()[0]
                    print unicode(id_cell),"-", unicode(value_cell), " |",
                else:
                    value_cell = x
                    print unicode(value_cell), " |",
        print
    return 0

def update_loc_relation(cell1, cell2, name_cell1, name_cell2):
    a = cell1['location']
    b = cell2['location']
    if ((a[1] <= b[1]) and (b[3] <= a[3])) or ((b[1] <= a[1]) and (a[3] <= b[3])):
        if 0 <= b[0] - a[2] <= MAX_LINE_Y:
            cell1["adjacent"]['down'].append(name_cell2)
            cell2["adjacent"]['up'].append(name_cell1)
        elif 0 <= a[0] - b[2] <= MAX_LINE_Y:
            cell1["adjacent"]['up'].append(name_cell2)
            cell2["adjacent"]['down'].append(name_cell1)
        return
    if ((a[0] <= b[0]) and (b[2] <= a[2])) or ((b[0] <= a[0]) and (a[2] <= b[2])):
        if 0 <= b[1] - a[3] <= MAX_LINE_Y:
            cell1["adjacent"]['right'].append(name_cell2)
            cell2["adjacent"]['left'].append(name_cell1)
        elif 0 <= a[1] - b[3] <= MAX_LINE_Y:
            cell1["adjacent"]['left'].append(name_cell2)
            cell2["adjacent"]['right'].append(name_cell1)
    return
# Export csv key value
# input file : location to export
# result_dict : result from detect key value
# debug = 0: 1 option
def export_csv(file,result_dict,debug=1):
    f = open(file, 'wt')
    try:
        writer = csv.writer(f, encoding='utf-8')
        #first = ('Key','Key_ID','Value') if debug == 0 else ('Key','Key_ID','Value_ID','Value')
        first = ('Key_ID', 'Value_ID')
        writer.writerow(first)
        other_list = []
        for key in result_dict.keys():
            if(key =='other'):
                other_list = result_dict['other']
                continue
            for i in range(len(result_dict[key])):
                row = []
                #row.append(result_dict[key][i]['value_key'].encode('utf-8'))
                row.append(result_dict[key][i]['id_key'])
                id_value_cells = []
                for x in result_dict[key][i]['value']:
                    if type(x) is dict:
                        id_cell = x.keys()[0]
                        id_value_cells.append(id_cell)
                        #value_cell = x.values()[0]
                row.append("|".join(id_value_cells))
                print(id_value_cells)
                        #row.append(value_cell)
                    #else:
                    #    value_cell = x
                    #    row.append(value_cell)
                row = [i.decode('utf-8') if type(i) is not unicode else i for i in row]
                writer.writerow(row)
        #writer.writerow(['Other'])
        for i in range(len(other_list)):
            row = []
            x = other_list[i]
            id_cell = x.keys()[0]
            value_cell = x.values()[0]
            row.append(id_cell)
            #row.append(value_cell)
            writer.writerow(row)
    finally:
        f.close()


    #print open(file, 'rt').read()



def detect_key_value(input_json,debug = 0):
    global output_json
    global key_value
    output_json = input_json
    key_value = constant.key_value
    table_keys = key_value.keys()
    ans_json = {}
    debug_json = {}
    other_list = []
    list_cells = sorted(output_json.keys())
    # Step 1 : Normalize clean up ocr_data 
    for cell in list_cells:
        output_json[cell]["value"] = normalize_output_text(output_json[cell]["value"])
        output_json[cell]["is_used"] = False
        output_json[cell]["adjacent"] = {"up": [], "down": [], "right": [], "left": []}
    print("INFO : KEY_VALUE print_table_cell after normalize")
    #print_table_cell(output_json)
    print("INFO : ---------len(output_json) = {}-----------------".format(len(output_json)))
    # Step 2:  Extract location relationship  
    for i, cell1 in enumerate(list_cells):
        for j in range(i):
            cell2 = list_cells[j]
            update_loc_relation(output_json[cell1], output_json[cell2], cell1, cell2)
    # Step 3: Update each cell is_used and value 
    for cell in list_cells:
        if output_json[cell]["is_used"]:
            continue
        text = output_json[cell]["value"]
        best_score = 0.0
        best_key = ""
        for cur_key in table_keys:
            cur_score = key_compare_score(cur_key, text)
            if cur_score > best_score:
                best_score = cur_score
                best_key = cur_key
        if best_score > 0.5:
            output_json[cell]["is_used"] = True
            output_json[cell]["best_key"] = best_key
    #Step 4:
    for cell in list_cells:
        if "best_key" in output_json[cell]:
            print("INFO : Processing cell : " + cell +" " + output_json[cell]["value"])
            if len(output_json[cell]["adjacent"]["right"]) > 1:
                num_neighbor_key = 0
                for l in output_json[cell]["adjacent"]["right"]:
                    if "best_key" in output_json[l]:
                        num_neighbor_key += 1
                if num_neighbor_key > 1 or num_neighbor_key == len(output_json[cell]["adjacent"]["right"]):
                    if len(output_json[cell]["adjacent"]["down"]) == 0:
                        continue
                    have_free_cell = False
                    for l in output_json[cell]["adjacent"]["down"]:
                        if not output_json[l]["is_used"]:
                            have_free_cell = True
                    if not have_free_cell:
                        continue
            best_key = output_json[cell]["best_key"]
            sv_out,sv_dbg = search_value(output_json[cell], key_value[best_key]["ValueTypes"],best_key.decode("utf-8") in keys_using_only_location,debug)
            dict_key = {'value_key': output_json[cell]["value"],
                        'id_key': cell,
                        'value': sv_out}
            dbg_key = {'value_key': output_json[cell]["value"],
                        'id_key': cell,
                        'value': sv_dbg}
            if best_key not in ans_json:
                ans_json[best_key] = [copy.deepcopy(dict_key)]
                debug_json[best_key] = [copy.deepcopy(dbg_key)]
            else:
                ans_json[best_key].append(copy.deepcopy(dict_key))
                debug_json[best_key].append(copy.deepcopy(dbg_key))
    #Step 5 : Track back not used cell
    for cell in list_cells:
        if not output_json[cell]["is_used"]:
            other_list.append({cell:output_json[cell]["value"]})
    print("INFO : -----------------------------------------")
    print("INFO : KEY_VALUE print out ans_json")
    print_result(debug_json)
    print("SUMMARY : Cannot map {} cells".format(len(other_list)))
    ans_json.update({"other":other_list})
    debug_json.update({"other":other_list})
    return ans_json,debug_json
def export_key_value(file,input_json):
    '''
# run key detect and export csv
# input : file -> export location
# input_json : raw json
# output_json : export debug json
    '''
    detected_data,debug_data = detect_key_value(input_json,debug = 1)
    print("SUMMARY : EXPORTING CSV file = {}".format(file))
    export_csv(file,detected_data)
    return detected_data

# Read json
def read_input(data):
    '''
    LOAD JSON in file_path format or pure json format
    '''
    raw_data = None
    try:
        input_file = file(data, "r")
        raw_data = json.loads(input_file.read().decode("utf-8"))
    except TypeError, e:
        print ("WARNING: Cannot run read_json assign raw_data = data")
        raw_data = data
    return raw_data



def mark_key_regions(result_dict, original, cell_cut, out_file_name):
    '''
    Draw key and value on original image
    input : self.debug_data
    input : self.raw_data
    output : marked image
    '''
    if len(original) == 2:
        original = (original * 255).astype('uint8')
        cv2.normalize(original, 0, 255, cv2.NORM_MINMAX)
        data2 = cv2.cvtColor(original, cv2.COLOR_GRAY2RGB)
    else:
        data2 = np.array(original, copy=True)
    list_key = []
    for key in result_dict.keys():
        if key == "other" :
            continue
        for i in range(len(result_dict[key])):
            cell_id = result_dict[key][i]['id_key']
            arr_location = cell_cut[cell_id]["location"]
            list_key.append({"cell_id": cell_id, "key": key, "ocr": cell_cut[cell_id]["value"]})
            p1 = (arr_location[1], arr_location[0])
            p2 = (arr_location[3], arr_location[2])
            cv2.rectangle(data2, p1, p2, (255, 0, 0), 3)
#            print out_file_name
            cv2.imwrite(out_file_name, data2)
            for x in result_dict[key][i]['value']:
                if type(x) is dict:
                    id_cell = x.keys()[0]
                    value_cell = x.values()[0]
                    arr_location = cell_cut[id_cell]["location"]
                    p1 = (arr_location[1], arr_location[0])
                    p2 = (arr_location[3], arr_location[2])
                    cv2.rectangle(data2, p1, p2, (0, 0, 255), 3)
                    cv2.imwrite(out_file_name, data2)
     #               print unicode(id_cell),"-", unicode(value_cell), " |",
                else:
                    print("ERROR: No table id found")
                    value_cell = x
     #               print unicode(value_cell), " |",

    return list_key

class KeyValueClassification:
    '''
    #KeyValueClassification class
    '''
    # config 
    raw_data =""
    processed_data =""
    debug_data =""
    # Default config
    confusion_matrix = ""
    config = {
        "debug": "0" ,
        "confusion":"0",
        "cm_path":"Key_detect_171220"
    }

    def __init__(self,cm_path = None):
        '''
        Basic skeleton
        init model , load config file
        '''
        print ("INFO : LOADING CONFUSION MATRIX DB")
        self.confusion_matrix = cm.load_cm(cm_path)
        return None

    def run_cm(self,input):
        '''
        correct input with confusion matrix
        '''
        fixed_data = cm.clean_up(input,self.confusion_matrix)

        return fixed_data


    def preprocess(self,data):
        ''' Step 1: Preprocessing / Clean up Data '''
        print("INFO : running preprocessing step..")
        # 1. Load in json data
        self.raw_data = data
        self.processed_data = copy.deepcopy(self.raw_data)
        # check running confusion 
        if self.config['confusion'] == "1":
            self.processed_data = cm.clean_up(self.processed_data,self.confusion_matrix)

        return 0

        # Step 2: Extract useful feature
    def feature_extraction(self):
        ''' Step 2: Preprocessing / Clean up Data '''
        print("INFO : running Feature Extraction step..")
        None
        '''
    def build_model(self):
        None
         Step 4 : Train
    def train_model(self):
        None
         Step 5 : Export model
    # LIST of model to be use in this section
    # any model should be write below
    # very basic linear regression to test dataset
    def model_linear_regression(self):
        None
        '''
    # logic model aka non_machine model to evaluate your algorithm
    def model_pure_logic(self):
        None

    def detect_kv(self,data,debug = 0):
        ''' MAIN TASK module detecting key value '''
        self.config["confusion"] = "0"
        self.preprocess(data)
        self.feature_extraction()
        ans, self.debug_data = detect_key_value(self.processed_data)
        return ans

    def detect_kv_cm(self,data,debug = 0):
        ''' MAIN TASK module detecting key value '''
        self.config["confusion"] = "1"
        self.preprocess(data)
        self.feature_extraction()
        ans, self.debug_data = detect_key_value(self.processed_data)
        return ans

    def dump_csv(self,file):
        '''Detect data to csv
        '''
        #detected_data = detect_key_value(data,debug = 1)
        print("SUMMARY : EXPORTING CSV file = {}".format(file))
        try:
            export_csv(file,self.debug_data)
        except:
            print ("ERROR: Please run detect_kv first")
        return self.debug_data



    def mark_key(self ,original_image, out_file_name):
        cell_cut = self.raw_data
        result_dict =self.debug_data
        list_keys = mark_key_regions(result_dict, original_image, cell_cut, out_file_name)
        return list_keys
'''
def main():
    model = KeyValueClassification()
    model.preprocess(raw)
'''
