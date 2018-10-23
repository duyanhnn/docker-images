# -*- coding: utf-8 -*-
import requests
import json


def ocr_space_file(filename, timeout=10, endpoint='https://api.ocr.space/parse/image', overlay=False, api_key='helloworld', language='jpn'):
    """ OCR.space API request with local file.
        Python3.5 - not tested on 2.7
    :param filename: Your file path & name.
    :param overlay: Is OCR.space overlay required in your response.
                    Defaults to False.
    :param api_key: OCR.space API key.
                    Defaults to 'helloworld'.
    :param language: Language code to be used in OCR.
                    List of available language codes can be found on https://ocr.space/OCRAPI
                    Defaults to 'en'.
    :return: Result in JSON format.
    """
    try:
        payload = {'isOverlayRequired': overlay,
                   'apikey': api_key,
                   'language': language,
                   }
        with open(filename, 'rb') as f:
            r = requests.post(endpoint,
                              files={filename: f},
                              data=payload,
                              timeout=timeout,
                              )

            resp = json.loads(r.content.decode('utf-8'))
            raw_parsed_text = resp['ParsedResults'][0]['ParsedText']
            parsed_text = raw_parsed_text.replace('\r\n', '')
    except Exception as e:
        print(e)
        parsed_text = ''
    return parsed_text
