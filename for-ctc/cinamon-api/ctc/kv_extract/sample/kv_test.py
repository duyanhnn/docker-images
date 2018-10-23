import os, sys
import argparse
import cv2
script_path = os.path.dirname(os.path.abspath( __file__ ))
sys.path.append(os.path.join(script_path,"../src"))
from extract_key_value import KeyValueClassification, ngram_models, read_input

# GRAB MAIN MODULE
if __name__ == '__main__':
    try:
        KV_DATA = os.environ["KV_DATA"]
        KV_TEST = os.environ["KV_TEST"]
        KV_PRJ  = os.environ["KV_PRJ"]
        print KV_DATA
    except KeyError:
        print "ERROR: Please set the environment variable KV_PRJ: Try source ./setup_env at PRJ/script folder"

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--test",
                        help="Specify test name, default: %(default)s",
                        default="test_demo")  # SINGLE TEST 
    args = parser.parse_args()
    reload(sys)
    sys.setdefaultencoding("utf-8")
   # parsing test 
    test_name = args.test
    test_dir = os.path.join(KV_TEST,test_name)
    print("INFO : START RUNNING TEST test_name = " + test_name)
    file_name = "original.png"
    ocr_out= os.path.join(test_dir, "data2.json")
    #input_json = read_json(os.path.join(test_dir, "data2.json"))
    ngram_models.init_model()
    #cm_file = '../cache/confusion/Key_detect_171220.csv'
    cm_file= '../cache/confusion/Key_detect_171220.csv'
    kv_clf = KeyValueClassification(cm_file)
    existed_data = read_input(ocr_out)
    #CM_DATA = data fixed but not key value detect
    cm_out = kv_clf.run_cm(existed_data)
    #kv_out = data run key detect  
    kv_out = kv_clf.detect_kv_cm(existed_data)
    # dump key value to csv file after run detect_kv
    kv_clf.dump_csv(test_dir + '/debug.csv')
    #ans = detect_key_value(input)
    #ans = export_key_value(test_dir  + '/debug.csv',input_json)
    image = cv2.imread(os.path.join(test_dir, file_name))
    out_file_name = os.path.join(test_dir, "key_marks.png")
    list_keys = kv_clf.mark_key( image , out_file_name)
    #out_xls_file = os.path.join(dir_name, file_name[:-4])
    #
