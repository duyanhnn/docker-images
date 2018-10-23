from __future__ import print_function
from __future__ import division
import cv2
import sys


def clahe_image(img_bgr, debug=False):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    if debug:
        return lab, l, a, b, cl, limg, final
    else:
        return final

if __name__ == "__main__":
    from matplotlib import pyplot as plt

    # -----Reading the image-------------------------------------------------
    img = cv2.imread(sys.argv[1], 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure()
    plt.title("original")
    plt.imshow(img_rgb)

    # -----Converting image to LAB Color model-------------------------------
    lab, l, a, b, cl, limg, final = clahe_image(img_rgb, debug=True)
    plt.figure()
    plt.title("lab")
    plt.imshow(lab)

    # -----Splitting the LAB image to different channels---------------------
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.title('l_channel')
    plt.gray()
    plt.imshow(l)
    plt.subplot(1, 3, 2)
    plt.gray()
    plt.title('a_channel')
    plt.imshow(l)
    plt.subplot(1, 3, 3)
    plt.gray()
    plt.title('b_channel')
    plt.imshow(l)

    # -----Applying CLAHE to L-channel---------------------------------------
    plt.figure()
    plt.gray()
    plt.title('CLAHE output')
    plt.imshow(cl)

    # -----Merge the CLAHE enhanced L-channel with the a and b channel-------
    plt.figure()
    plt.title('limg')
    plt.imshow(limg)

    # -----Converting image from LAB Color model to RGB model----------------
    plt.figure()
    plt.title('final')
    plt.imshow(final)
    plt.show()
    cv2.imwrite("out_clare.jpg", final)
