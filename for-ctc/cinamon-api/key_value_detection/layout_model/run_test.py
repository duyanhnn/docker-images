from __future__ import print_function, division

import click
from .layout_model import LayoutModel
from .util import read_image_list

@click.command()
@click.option('--path_list_imgs', default="./demo_images/imgs.lst")
@click.option('--path_net_border', default="/home/taprosoft/Downloads/test_segmented/flax_bprost/run/code/ARU-Net/frozen_model_border_epoch200.pb")
@click.option('--path_net_text', default="/home/taprosoft/Downloads/test_segmented/flax_bprost/run/code/ARU-Net/frozen_model_server.pb")
@click.option('--export_dir', default="./demo_images")
@click.option('--scale', type=float, default=0.35)

def run(path_list_imgs, scale, path_net_border, path_net_text, export_dir):
    list_inf = read_image_list(path_list_imgs)
    inference = LayoutModel(path_net_text, mode='L', scale=scale, list_path_to_pb_alt=[path_net_border])
    inference.inference_list(list_inf, export_dir=export_dir, show_debug_text=False, write_lines=False, show_border=True)

if __name__ == '__main__':
    run()