import os
import json
import requests
import uuid

def read_file(filename):
    with open(filename, 'rb') as f:
        img = f.read()
    return img


def test_flax():
    input_file = "input_6.pdf"
    img_blob = read_file(input_file)
    payload = {
        "request_id": str(uuid.uuid1())[0:31],
    }
    api = "http://52.243.58.135:8000/api/v1/upload/"
    r = requests.post(api,
                      data=payload,
                      files={"image": (os.path.basename(input_file), img_blob)},
                      )
    resp = json.loads(r.content.decode('utf-8'))
    print(resp)


test_flax()
