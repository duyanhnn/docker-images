from ..bboxs_tools.bbox_operations import get_x_y_start_stop
from PIL import ImageFont
import cv2


def draw_cover_rectangle(img, rects, output):
    new_img = img.copy()
    for rect in rects:
        new_img = cv2.drawContours(new_img, [rect], 0, (0, 0, 255), 2)
    cv2.imwrite(output, new_img)


def draw_cover_rectangle_with_types(img, rects, types, output):
    new_img = img.copy()
    for i, rect in enumerate(rects):
        if types[i] == 0:
            color = (0, 0, 255)
        elif types[i] > 2 and types[i] != 0:
            color = (0, 255, 0)
        else:
            color = (255, 0, 0)
        new_img = cv2.drawContours(new_img, [rect], 0, color, 2)
    cv2.imwrite(output, new_img)


def draw_text_in_a_box(draw, text, bbox):
    """
    Args:
        draw: a PIL draw
    """
    fontsize = 1  # starting font size
    x_start, x_stop, y_start, y_stop = get_x_y_start_stop(bbox)
    width = x_stop - x_start
    height = y_stop - y_start

    # portion of image width you want text width to be

    font = ImageFont.truetype("fonts-japanese-gothic.ttf", fontsize)
    while font.getsize(text)[0] < width and font.getsize(text)[1] < height:
        # iterate until the text size is just larger than the criteria
        fontsize += 1
        font = ImageFont.truetype("fonts-japanese-gothic.ttf", fontsize)

    # optionally de-increment to be sure it is less than criteria
    fontsize -= 1
    font = ImageFont.truetype("fonts-japanese-gothic.ttf", fontsize)

    draw.text((x_start, y_start), text, font=font)
