import json
import subprocess
from wand.image import Image, Color
import cv2

PDF2PNG_CMD = 'convert -quality 100 -density 150 -background white -alpha remove -append {} {}'


def read_json_file(file):
    with open(file) as f:
        data = f.read()
    return json.loads(data)


def dump_to_file(data, file):
    with open(file, 'w') as fp:
        json.dump(data, fp)


def pdf_to_image(pdf_file, image_file):
    command = PDF2PNG_CMD.format(pdf_file, image_file)
    command = command.split()
    exit_code = 0
    output = ''
    try:
        output = subprocess.check_output(command, shell=False, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        exit_code = e.returncode
    if 'error' in str(output).lower():
        raise Exception
    return image_file


def read_pdf_image(pdf_file_path, output_img=None):
    """Convert pdf to image png"""
    out_path = output_img
    with Image(filename=pdf_file_path, resolution=300) as pdf:
        pages = len(pdf.sequence)
        if pages > 1:
            image = Image(
                width=pdf.width,
                height=pdf.height * pages
            )
            for i in range(pages):
                image.composite(
                    pdf.sequence[i],
                    top=pdf.height * i,
                    left=0
                )
        else:
            image = pdf
        image.background_color = Color("white")
        image.alpha_channel = 'remove'
        image.save(filename=out_path)
        new_image = cv2.imread(out_path)
    return new_image