from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from matplotlib import pyplot as plt


def plot_img(img, is_gray=False,
             new_figure=False, show_now=False,
             plt_title=""):
    """Plot image
    Args:
        img: the image, numpy array
        is_gray: to plot gray or not
        new_figure: create new figure or not
        show_now: show immediately after func
        plt_title: the axes's title
    """
    if new_figure:
        plt.figure()
    if is_gray:
        plt.gray()
    plt.imshow(img)
    if plt_title:
        plt.title(plt_title)
    if show_now:
        plt.show()


def plot_gray(img, new_figure=False, show_now=False, plt_title=""):
    """Plot image in gray
    Args:
        img: the image, numpy array
        new_figure: create new figure or not
        show_now: show immediately after func
        plt_title: the axes's title
    """
    plot_img(img, True, new_figure, show_now, plt_title)
