import os
import json

from PIL import Image
import requests
import pytesseract

def run_tesseract(image):
    pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract4'
    ocr_value = ''
    ocr_value = pytesseract.image_to_string(Image.open(image), lang='jpn')
    ocr_value = ocr_value.strip()
    if isinstance(ocr_value, str):
        ocr_value = ocr_value.decode('utf-8')
    return ocr_value

def ocr_request(filename, endpoint, timeout=10):
    fn = os.path.basename(filename).split('.')[0]
    try:
        with open(filename, 'rb') as f:
            r = requests.post(endpoint,
                              files={'image': f},
                              timeout=timeout,
                              )
            print(r.content)
            resp = json.loads(r.content.decode('utf-8'))
            ocr_value = resp['data']['result']
            data = {fn: ocr_value}
    except Exception as e:
        print(e)
        data = {fn: ''}
    print(data)
    return data