import os
import re


def get_all_file(path, ext):
    list_files = []
    for root, dires, files in os.walk(path):
        for the_file in files:
            if the_file.endswith(ext):
                list_files.append(os.path.join(root, the_file))
    return list_files


def get_all_file_and_rename(path, ext):
    list_files = []
    for root, dires, files in os.walk(path):
        for the_file in files:
            if the_file.endswith(ext):
                the_new_file = re.sub(r'\s+', '', the_file)
                os.rename(os.path.join(root, the_file),
                          os.path.join(root, the_new_file))
                list_files.append(os.path.join(root, the_new_file))
    return list_files


def make_sure_dir_exists(dir_name):
    if not os.path.exists(dir_name):
        dir_name = os.path.join(os.getcwd(),
                                dir_name)
        os.makedirs(dir_name)
    return dir_name
