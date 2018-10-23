import os
import glob
import json
import argparse
import re


parser = argparse.ArgumentParser(description="Convert extention of image")
parser.add_argument("-f", "--folder", help="root folder path of images")
parser.add_argument("-o", "--output", help="output format of image", default='png')
parser.add_argument("-i", "--input", help="input format of image")
args = parser.parse_args()  


def rename_file(file_name, index, convert_path, data_result):
    index = index + 1
    file_name = '0'*(5-len(str(index))) + str(index) +'.png'
    if os.path.isfile(convert_path):
        named = name_existed(file_name, data_result)
        while named:
            index = index+1
            file_name = '0'*(5-len(str(index))) + str(index)+ '.png'
            named = name_existed(file_name,data_result)
    return file_name

def is_existed(img_path, data_result):
    is_check = False
    for k, v in data_result.items():
        if os.path.basename(img_path) == k:
            is_check = True
    return is_check

def name_existed(file_name, data_result):
    is_check = False
    for k, v in data_result.items():
        if file_name == v:
            is_check = True
    return is_check 

def main():
    folder_img = glob.glob(os.path.join(args.folder,'*.{}*'.format(args.input)))
    data_result = {}
    is_converted = False
    convert_path = os.path.join(os.path.dirname(args.folder), 'convert.json')
    if os.path.isfile(convert_path):
        is_converted = True
        with open(convert_path,'r',encoding='utf-8') as f:
            data_result = json.load(f)
    print(data_result)
    for index, img_path in enumerate(sorted(folder_img)):
        existed = False
        if is_converted:
            existed = is_existed(img_path, data_result)
        if existed: 
            print('Converted previously!')
            continue
        else:
            file_name = rename_file(os.path.basename(img_path), index, convert_path, data_result)
            output = os.path.join(os.path.dirname(img_path), file_name) 
            os.system("convert '{}[0]' -quality 100 {}".format(img_path, output))
            print('Converted {}'.format(file_name))
            data_result[os.path.basename(img_path)] = file_name
    with open(convert_path,'w',encoding='utf-8') as f:
        f.write(json.dumps(data_result, ensure_ascii=False))
    print(data_result)
    print('-----Done-----')


if __name__ == '__main__':
    main()
