# coding=utf-8
import os , sys
import csv
import codecs
import re

def normalize_output_text(text):
    #text = unicode(text)
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


def load_cm(file):
    ''' convert csv to dict
    path: path to confusion _matrix dir
    '''
    dict_list = []
    csv_list = []
    csv_list.append(file)
    print csv_list
    '''
    CONFUSION_DIR = path
    # get all consiion csv files
    for file in sorted(os.listdir(CONFUSION_DIR)):
        if file.endswith(".csv") and not file.startswith("~"):
            csv_list.append(file)
    '''
    for file in csv_list:
        #reader = csv.DictReader(open(CONFUSION_DIR + "/" + file,'rb'))
        reader = csv.DictReader(open(file,'rb'))
        for line in reader:
            dict_list.append(line)
#    print("INFO : size of confusion matrix = {}".format(len(dict_list)))
    return dict_list

def find_fix(v_in,id_in,confusion_matrix):
    for i in confusion_matrix:
        v = i["value"]
        id = i["id"]

        if normalize_output_text(v_in) == normalize_output_text(v) :
            if i["truth"] != 'True':
                v_out = i["truth"]
            else:
                v_out = "Not match"
            return v_out
    return "Not match"

def clean_up(data,cm_list):
    '''
    LOAD CSV confusion file
    '''
    print ("INFO : USING CONFUSION DB TO CORRECT OCR RESULT")
    dict_list = cm_list
    '''
    for i in dict_list:
        print i["id"], normalize_output_text(i["value"])
        '''
    cell_list = sorted(data.keys())
    for cell in  cell_list:
        v = data[cell]["value"]
        id = cell
        if(v != " "):
            #print id , normalize_output_text(v)
            fix = find_fix(v,id,dict_list)
            if( fix != "Not match"):
                data[cell]["value"] = unicode(fix)
                print("INFO : CF_MATRIX "+ id + " " +  v + " correct to ->  " + data[cell]["value"])
    return data
