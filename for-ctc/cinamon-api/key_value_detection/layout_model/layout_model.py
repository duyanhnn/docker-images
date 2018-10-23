from __future__ import print_function, division

import time

# import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import shutil
from scipy import misc
import cv2
import os
from .util import load_graph, resize_to_prefered_width, resize_to_prefered_height
from .visualize import visualize_line_seg_hires_area, visualize_border_hires, export_lines_to_disk, \
    threshold_and_upscale_all_channel, draw_mask, draw_rects, export_lines, scale_rect, skelet, threshold_and_upscale_map
from keras.preprocessing import image

from .morph import assign_line_to_cell
from .rotate_license import rotate_license, read_color_license
import keras

COUNT_TIME = True

class ImportGraph():
    """  Importing and running isolated TF graph """

    def __init__(self, path_to_pb, gpu_device="0", input_tensor_name="inImg", output_tensor_name="output"):
        # Create local graph and use it in the session
        self.graph = load_graph(path_to_pb)
        self.gpu_device = gpu_device

        self.x = self.graph.get_tensor_by_name('{}:0'.format(input_tensor_name))
        self.predictor = self.graph.get_tensor_by_name('{}:0'.format(output_tensor_name))


    def run(self, data):
        """ Running the activation operation previously imported """
        # The 'x' corresponds to name of input placeholder
        session_conf = tf.ConfigProto(use_per_session_threads=True)
        session_conf.gpu_options.visible_device_list = self.gpu_device
        self.sess = tf.Session(graph=self.graph, config=session_conf)
        result =  self.sess.run(self.predictor, feed_dict={self.x: data})
        self.sess.close()
        return result

    def close(self):
        self.sess.close()


class LayoutModel(object):
    """
        Perform inference for an arunet instance

        :param net: the arunet instance to train

        """
    def __init__(self, path_to_pb, scale=0.4, mode='L', list_path_to_pb_alt = None, load_license_model = False, license_model_data_paths = None):
        self.graph = ImportGraph(path_to_pb)

        if list_path_to_pb_alt is not None:
            self.graph_alt = []
            for path in list_path_to_pb_alt[:1]:
                self.graph_alt.append(ImportGraph(path))
        else:
            self.graph_alt = None

        if load_license_model and license_model_data_paths is not None:
            # self.license_model = TransferModel(final_layer_weight=license_model_data_paths[-2], class_labels_path=license_model_data_paths[-1])
            import json
            self.class_labels = json.load(open(license_model_data_paths[-1], "r"))
            self.img_shape = (128, 128)
            self.graph_alt.append(ImportGraph(list_path_to_pb_alt[1], input_tensor_name='input_1', output_tensor_name='output_node0'))
            self.graph_alt.append(ImportGraph(list_path_to_pb_alt[2], input_tensor_name='input_2', output_tensor_name='output_node0'))

        self.scale = scale
        self.mode = mode

    def process_one_file(self, aImgPath, export_dir, write_lines=True, show_debug_text=True, show_border=False):
        file_name = os.path.splitext(os.path.basename(aImgPath))[0]
        print(
            "Image: {:} ".format(aImgPath))
        print(file_name)

        batch_x = self.load_img(aImgPath, self.scale, self.mode, with_flip=False)
        # color_im = self.load_img(aImgPath, self.scale, "RGB")

        origin_im_normal_size = cv2.imread(aImgPath)
        origin_im = cv2.imread(aImgPath, cv2.IMREAD_REDUCED_COLOR_2)
        origin_im = self.resize_to_preferred_dim(origin_im)
        export_scale_factor = 1.0 * origin_im_normal_size.shape[0] / origin_im.shape[0]

        print(
            "Resolution: h {:}, w {:} ".format(batch_x.shape[1], batch_x.shape[2]))

        hires_outs = []

        aTime = time.time()
        # Run validation
        for graph, line_seg_mode in [(self.graph, 1), (self.graph_alt[0], 0)]:  # ,(self.graph_alt[1], 2) ]:
            aPred = graph.run(batch_x)
            n_class = aPred.shape[3]
            channels = batch_x.shape[3]

            im_shape = aPred[0, :, :, 1].shape

            if line_seg_mode == 1:
                mask = np.dstack((np.zeros(im_shape, dtype='uint8'), 1 - aPred[0, :, :, 2],
                                  np.zeros(im_shape, dtype='uint8'))) * 255
                # misc.imsave(file_name + '_line_blend.jpg', 0.5 * color_im + 0.5 * mask)

                hires_outs.append((aPred[0, :, :, 0] * 255).astype('uint8'))
                hires_outs.append((aPred[0, :, :, 1] * 255).astype('uint8'))
                hires_outs.append((aPred[0, :, :, 2] * 255).astype('uint8'))
            elif line_seg_mode == 0:
                mask = np.dstack(
                    (np.zeros(im_shape, dtype='uint8'), aPred[0, :, :, 1], np.zeros(im_shape, dtype='uint8'))) * 255
                # misc.imsave(file_name + '_border_blend.jpg', 0.5 * color_im + 0.5 * mask)

                hires_outs.append((aPred[0, :, :, 1] * 255).astype('uint8'))
        hires_outs = threshold_and_upscale_all_channel(hires_outs, origin_im.shape)

        if COUNT_TIME:
            curTime = (time.time() - aTime) * 1000.0
            aTime = time.time()
            print(
                "Model predict time: {:.2f} ms".format(curTime))

        new_box = True
        if not new_box:
            debug_line, line_mask, lines = visualize_line_seg_hires_area(origin_im, hires_outs[0], hires_outs[1],
                                                                         hires_outs[2], hires_outs[3], new_box=new_box)
        else:
            debug_line, line_mask, lines, lines_with_box = visualize_line_seg_hires_area(origin_im, hires_outs[0], hires_outs[1],
                                                                         hires_outs[2], hires_outs[3], new_box=new_box)
        if COUNT_TIME:
            curTime = (time.time() - aTime) * 1000.0
            aTime = time.time()
            print(
                "Line seg time: {:.2f} ms".format(curTime))

        debug_cell, cell_mask, cells = visualize_border_hires(origin_im, hires_outs[3], hires_outs[2], show_text=show_debug_text)

        line_cell_id, blank = assign_line_to_cell(lines, cells, origin_im.shape[:2], show_text=show_debug_text)

        if COUNT_TIME:
            curTime = (time.time() - aTime) * 1000.0
            aTime = time.time()
            print(
                "Cell cut time: {:.2f} ms".format(curTime))

        if write_lines:
            cv2.imwrite(os.path.join(export_dir, '_border_comp.jpg'), debug_cell)
        else:
            cv2.imwrite(os.path.join(export_dir, '{}_border_comp.jpg').format(file_name), debug_cell)

        hires_combined = debug_line
        hires_combined = cv2.addWeighted(origin_im, 0.7, hires_combined, 0.3, 0)

        hires_combined_mask = hires_combined.copy()
        hires_combined_mask[line_mask > 0] = [0, 0, 255]
        hires_combined_mask[blank > 0] = [0, 0, 0]

        hires_combined = cv2.addWeighted(hires_combined_mask, 0.5, hires_combined, 0.5, 0)
        if show_border:
            cell_mask = hires_outs[3]
        hires_combined_alt = draw_mask(hires_combined, hires_outs[3], [0,255,0], 0.65)
        # hires_combined_alt = draw_mask(hires_combined, cell_mask, [0,255,0], 0.75)

        if write_lines:
            cv2.imwrite(os.path.join(export_dir, '_comp.jpg'), hires_combined_alt)
            # cv2.imwrite(os.path.join(export_dir, '_comp_alt.jpg'), hires_combined_alt)
        else:
            cv2.imwrite(os.path.join(export_dir, '{}_comp.jpg'.format(file_name)), hires_combined_alt)
            # cv2.imwrite(os.path.join(export_dir, '{}_comp_alt.jpg'.format(file_name)), hires_combined_alt)

        if write_lines:
            print("Exporting line images...")
            if not new_box:
                export_lines_to_disk(origin_im_normal_size, lines, cells, line_cell_id, dir_name=export_dir, scale_factor=export_scale_factor)
            else:
                export_lines_to_disk(origin_im_normal_size, lines_with_box, cells, line_cell_id, dir_name=export_dir, scale_factor=export_scale_factor, new_box=True)

        if COUNT_TIME:
            curTime = (time.time() - aTime) * 1000.0
            aTime = time.time()
            print(
                "Export time: {:.2f} ms".format(curTime))


    def process_one_file_with_line_only(self, aImgPath, cells, export_dir, return_line_im=False, return_debug_im=True, safe_segmentation=True, new_box=True):
        file_name = os.path.splitext(os.path.basename(aImgPath))[0]
        print(
            "Image: {:} ".format(aImgPath))
        print(file_name)

        batch_x = self.load_img(aImgPath, self.scale, self.mode, with_flip=False)

        origin_im_normal_size = cv2.imread(aImgPath)
        origin_im = cv2.imread(aImgPath, cv2.IMREAD_REDUCED_COLOR_2)
        origin_im = self.resize_to_preferred_dim(origin_im)
        export_scale_factor = 1.0 * origin_im_normal_size.shape[0] / origin_im.shape[0]

        cell_mask = np.zeros(origin_im_normal_size.shape[:2], dtype='uint8')
        draw_rects(cell_mask, cells, 255, 2)

        print(
            "Resolution: h {:}, w {:} ".format(batch_x.shape[1], batch_x.shape[2]))

        hires_outs = []

        # Run validation
        for graph, line_seg_mode in [(self.graph, 1)]:  # ,(self.graph_alt[1], 2) ]:
            aPred = graph.run(batch_x)

            im_shape = aPred[0, :, :, 1].shape

            if line_seg_mode == 1:
                # mask = np.dstack((np.zeros(im_shape, dtype='uint8'), 1 - aPred[0, :, :, 2],
                #                   np.zeros(im_shape, dtype='uint8'))) * 255
                # misc.imsave(file_name + '_line_blend.jpg', 0.5 * color_im + 0.5 * mask)

                hires_outs.append((aPred[0, :, :, 0] * 255).astype('uint8'))
                hires_outs.append((aPred[0, :, :, 1] * 255).astype('uint8'))
                hires_outs.append((aPred[0, :, :, 2] * 255).astype('uint8'))

        hires_outs.append(cell_mask)

        hires_outs = threshold_and_upscale_all_channel(hires_outs, origin_im.shape)

        if not new_box:
            debug_line, line_mask, lines = visualize_line_seg_hires_area(origin_im, hires_outs[0], hires_outs[1],
                                                                         hires_outs[2], hires_outs[3], safe_segmentation=safe_segmentation)
        else:
            debug_line, line_mask, lines, lines_with_box = visualize_line_seg_hires_area(origin_im, hires_outs[0],
                                                                                         hires_outs[1],
                                                                                         hires_outs[2], hires_outs[3],
                                                                                         new_box=new_box)

        normalized_lines = [scale_rect(l, export_scale_factor) for l in lines]

        line_cell_id, line_id_mask = assign_line_to_cell(normalized_lines, cells, origin_im.shape[:2])

        if return_line_im:
            if new_box:
                return export_lines(origin_im_normal_size, lines_with_box, cells, line_cell_id, scale_factor=export_scale_factor, new_box=new_box)
            else:
                return export_lines(origin_im_normal_size, lines, cells, line_cell_id,
                                    scale_factor=export_scale_factor, new_box=new_box)

        if return_debug_im:
            hires_combined = debug_line
            hires_combined = cv2.addWeighted(origin_im, 0.7, hires_combined, 0.3, 0)

            hires_combined_mask = hires_combined.copy()
            hires_combined_mask[line_mask > 0] = [0, 0, 255]
            hires_combined_mask[line_id_mask > 0] = [0, 0, 0]

            hires_combined = cv2.addWeighted(hires_combined_mask, 0.4, hires_combined, 0.6, 0)
            draw_rects(hires_combined, cells, [0, 255, 0], 2)
            if export_dir is not None:
                cv2.imwrite(os.path.join(export_dir, '{}_comp.jpg').format(file_name), hires_combined)

            return lines, line_cell_id, hires_combined

        return lines, line_cell_id


    def warp_license(self, aImgPath, export_dir = None, debug_name = False):
        file_name = os.path.splitext(os.path.basename(aImgPath))[0]
        print(
            "Image: {:} ".format(aImgPath))
        print(file_name)

        batch_x = self.load_img(aImgPath, self.scale, self.mode, with_flip=False)
        origin_im_normal_size = cv2.imread(aImgPath)
        origin_im = self.resize_to_preferred_dim_license(origin_im_normal_size.copy())

        print(
            "Resolution: h {:}, w {:} ".format(batch_x.shape[1], batch_x.shape[2]))

        gt_border = None
        h, w = origin_im.shape[:2]

        # Run validation
        for graph, line_seg_mode in [(self.graph_alt[0], 2)]:
            aPred = graph.run(batch_x)
            gt_border = (aPred[0, :, :, 1] * 255).astype('uint8')

        gt_border = cv2.resize(gt_border, (w, h))
        gt_border = (skelet(gt_border, thres=100, expand=False, iter=2) * 255).astype('uint8')
        origin_im[gt_border > 100] = [0, 255, 0]

        origin_warped_out, border_warped, output_cell_checked, output_cell_ims, content_mask, debug_ims = rotate_license(gt_border, origin_im, origin_im_normal_size, filename=None)

        warped_file_name = None

        cell_ims_checked = []
        cell_ims_checked_indexes = []

        for i, cell_im in enumerate(output_cell_ims):
            if output_cell_checked[i]:
                cell_ims_checked.append(cell_im)
                cell_ims_checked_indexes.append(i)
                # cv2.imwrite(os.path.join(export_dir, '{}_cell_{}.jpg'.format(file_name, i)), cell_im)

        output_labels_checked = self.read_license_cell(cell_ims_checked, cell_ims_checked_indexes)

        if export_dir is not None:
            if debug_name:
                # cv2.imwrite(os.path.join(export_dir, '{}_border_warped.jpg'.format(file_name)), border_warped)
                # cv2.imwrite(os.path.join(export_dir, '{}_debug_im.jpg'.format(file_name)), debug_ims[1])
                # cv2.imwrite(os.path.join(export_dir, '{}_debug_warped.jpg'.format(file_name)), debug_ims[0])
                warped_file_name = os.path.join(export_dir, '{}_warped.png'.format(file_name))
                # cv2.imwrite(warped_file_name, origin_warped_out)
            else:
                cv2.imwrite(os.path.join(export_dir, '_border_warped.jpg'), border_warped)
                cv2.imwrite(os.path.join(export_dir, '_debug_im.jpg'), debug_ims[1])
                cv2.imwrite(os.path.join(export_dir, '_debug_warped.jpg'), debug_ims[0])
                warped_file_name = os.path.join(export_dir, '_warped.png')
                cv2.imwrite(warped_file_name, origin_warped_out)

        return warped_file_name, content_mask, debug_ims[0], output_labels_checked

    def predict_license_cells(self, imgs, resize=False, reverse_channel=False):
        transfer_model = self.graph_alt[1]
        final_model = self.graph_alt[2]

        img_arr = []
        if resize:
            for im in imgs:
                im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                im = cv2.resize(im, self.img_shape)
                im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
                if reverse_channel:
                    im = im[:, :, ::-1]
                img_arr.append(image.img_to_array(im))
        img_arr = np.stack(img_arr, axis=0)
        preprocess_input = keras.applications.mobilenet.preprocess_input
        img_arr = preprocess_input(img_arr)

        print("predicting license cells...")

        x_features = transfer_model.run(img_arr)
        y_proba = final_model.run(x_features)
        y_ind = np.argmax(y_proba, axis=-1)
        labels = [self.class_labels[str(y)] for y in y_ind]

        return labels

    def read_license_cell(self, cell_ims, indexes):
        default_map = {0:'大型', 1: '中型', 2:'普通', 3:'大特' ,4:'大自二', 5:'普自二', 6:'小特', 7:'原付', 8:'大特二', 9:'大二', 10:'中二', 11:'普二' ,12:'大特二' ,13:'け引二'}
        results = []
        need_model_read_ims = []
        for i, cell_im in enumerate(cell_ims):
            if indexes[i] in range(9, 14):
                results.append(default_map[indexes[i]])
            else:
                need_model_read_ims.append(cell_im)

        if len(need_model_read_ims):
            labels = self.predict_license_cells(need_model_read_ims, resize=True, reverse_channel=False) #self.license_model.predict_ims(need_model_read_ims, resize=True, reverse_channel=True)
        else:
            labels = []
        results += labels
        print("cell checked in license: ", results)
        return results

    def process_warped_file_license(self, aImgPath, cells, output_cell_checked, content_mask, export_dir, write_lines=True, debug_im = None):
        file_name = os.path.splitext(os.path.basename(aImgPath))[0]
        print(
            "Image: {:} ".format(aImgPath))
        print(file_name)

        batch_x = self.load_img(aImgPath, self.scale, self.mode, with_flip=False)

        origin_im_normal_size = cv2.imread(aImgPath)
        origin_im = cv2.imread(aImgPath, cv2.IMREAD_REDUCED_COLOR_2)
        origin_im = self.resize_to_preferred_dim(origin_im)
        if debug_im is None:
            debug_im = origin_im
        else:
            debug_im = cv2.resize(debug_im, (origin_im.shape[1], origin_im.shape[0]))

        export_scale_factor = 1.0 * origin_im_normal_size.shape[0] / origin_im.shape[0]

        cell_mask = np.zeros(origin_im_normal_size.shape[:2], dtype='uint8')
        draw_rects(cell_mask, cells, 255, 2)

        print(
            "Resolution: h {:}, w {:} ".format(batch_x.shape[1], batch_x.shape[2]))

        hires_outs = []

        # Run validation
        for graph, line_seg_mode in [(self.graph, 1)]:  # ,(self.graph_alt[1], 2) ]:
            aPred = graph.run(batch_x)
            hires_outs.append((aPred[0, :, :, 0] * 255).astype('uint8'))
            hires_outs.append((aPred[0, :, :, 1] * 255).astype('uint8'))
            hires_outs.append((aPred[0, :, :, 2] * 255).astype('uint8'))

        hires_outs.append(cell_mask)

        hires_outs = threshold_and_upscale_all_channel(hires_outs, origin_im.shape)
        content_mask = threshold_and_upscale_map(origin_im.shape, content_mask, threshold=0)

        # cv2.imwrite(os.path.join(export_dir, '{}_mask.jpg').format(file_name), (content_mask * 255).astype('uint8'))

        for i in [0,1,2]:
            hires_outs[i][content_mask > 0] = 0

        debug_line, line_mask, lines = visualize_line_seg_hires_area(origin_im, hires_outs[0], hires_outs[1],
                                                                     hires_outs[2], hires_outs[3], safe_segmentation=True)

        line_cell_id, line_id_mask = assign_line_to_cell(lines, cells, origin_im.shape[:2])

        color_id, pos = read_color_license(lines, origin_im)
        color_map = {0: 'gold', 1: 'blue', 2: 'green', -1: 'error'}
        color_map_rgb = {0:(134,169,196), 1:(189,166,88), 2:(79,246,179)}

        if write_lines:
            print("Exporting line images...")
            checked_cell_json = {"checked_cells":[]}
            for i, is_checked in enumerate(output_cell_checked):
                # if is_checked:
                checked_cell_json["checked_cells"].append(is_checked)
            checked_cell_json["color"] = color_map[color_id]
            export_lines_to_disk(origin_im_normal_size, lines, cells, line_cell_id, dir_name=export_dir,
                                 scale_factor=export_scale_factor, addition_json=checked_cell_json)

        hires_combined = debug_line
        hires_combined = cv2.addWeighted(origin_im, 0.5, hires_combined, 0.5, 0)

        hires_combined_mask = hires_combined.copy()
        hires_combined_mask[line_mask > 0] = [0, 0, 255]
        hires_combined_mask[line_id_mask > 0] = [0, 0, 0]
        if color_id >= 0:
            cv2.putText(hires_combined_mask, color_map[color_id].upper(), (pos[0] - 100, pos[1] + 30), color=color_map_rgb[color_id],
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0, thickness=3)

        hires_combined = cv2.addWeighted(hires_combined_mask, 0.8, hires_combined, 0.2, 0)
        debug_im = cv2.addWeighted(debug_im, 0.4, hires_combined, 0.6, 0)
        # draw_rects(hires_combined, cells, [0, 255, 0], 2)
        if write_lines:
            cv2.imwrite(os.path.join(export_dir, '_comp.jpg'), debug_im)
        else:
            cv2.imwrite(os.path.join(export_dir, '{}_comp.jpg'.format(file_name)), debug_im)


    def process_license(self, aImgPath, export_dir = "./", write_lines=True):
        warped_file_name, content_mask, debug_im, output_cell_checked =  self.warp_license(aImgPath, export_dir=export_dir, debug_name=False)
        self.process_warped_file_license(warped_file_name, [], output_cell_checked, content_mask, export_dir=export_dir, write_lines=write_lines, debug_im=debug_im)


    def inference_list(self, img_list, export_dir, write_lines=False, show_debug_text=True, show_border=False, mode=0):
        val_size = len(img_list)
        if val_size is None:
            print("No Inference Data available. Skip Inference.")
            return

        print("Start Inference")
        for step in range(0, val_size):
            aImgPath = img_list[step]
            filename = os.path.split(aImgPath)[-1].split('.')[0]
            aTime = time.time()
            if mode == 0:
                if write_lines:
                    export_dir_file = os.path.join(export_dir,filename)
                    try:
                        os.mkdir(export_dir_file)
                    except:
                        shutil.rmtree(export_dir_file)
                        os.mkdir(export_dir_file)
                else:
                    export_dir_file = export_dir
                print('\nExport dir', export_dir_file)
                self.process_one_file(aImgPath, export_dir=export_dir_file, write_lines=write_lines,
                                      show_debug_text=show_debug_text, show_border=show_border)
            elif mode == 1:
                self.process_one_file_with_line_only(aImgPath, [], export_dir=export_dir, return_line_im=False)
            elif mode == 2:
                self.process_license(aImgPath, export_dir=export_dir, write_lines=write_lines)

            curTime = (time.time() - aTime) * 1000.0
            aTime = time.time()
            print(
                "Update time: {:.2f} ms".format(curTime))

        print("Inference Finished!")

        # finish and close tensorflow session
        # self.graph.close()
        # if self.graph_alt is not None:
        #     for graph in self.graph_alt:
        #         graph.close()

        return None

    def output_epoch_stats_val(self, time_used):
        print(
            "Inference avg update time: {:.2f} ms".format(time_used))

    def resize_to_preferred_dim(self, aImg, scale = 1.0):
        h, w = aImg.shape[:2]
        ratio = 1.0 * h / w
        if ratio > 3.6:
            aImg = resize_to_prefered_width(aImg, int(800 * scale))
        elif ratio > 2:
            aImg = resize_to_prefered_width(aImg, int(1200 * scale))
        elif ratio > 1.3:
            aImg = resize_to_prefered_width(aImg, int(2200 * scale))
        else:
            aImg = resize_to_prefered_height(aImg, int(1900 * scale))

        return aImg

    def resize_to_preferred_dim_license(self, aImg):
        h, w = aImg.shape[:2]

        if h > w * 1.3:
            aImg = resize_to_prefered_width(aImg, 1200)  # 2400
        else:
            aImg = resize_to_prefered_height(aImg, 800)  # 2200

        return aImg

    def load_img(self, path, scale, mode, with_flip=False):
        aImg = cv2.imread(path, cv2.IMREAD_GRAYSCALE) #misc.imread(path, mode=mode)
        aImg = self.resize_to_preferred_dim(aImg)

        if max(aImg.shape[:2]) > 2000:
            # if aImg.shape[0] < 4200:
            if 1.0 * aImg.shape[0] / aImg.shape[1] < 2.0:
                sImg = misc.imresize(aImg, scale, interp='bicubic')
            else:
                sImg = misc.imresize(aImg, scale * 1.5, interp='bicubic')
        elif max(aImg.shape[:2]) > 1300:
            sImg = misc.imresize(aImg, scale * 1.2, interp='bicubic')
        else:
            sImg = aImg
        fImg = sImg
        if len(sImg.shape) == 2:
            fImg = np.expand_dims(fImg,2)

        if with_flip:
            fImg = np.array((fImg, np.flip(fImg, 0)))
        else:
            fImg = np.expand_dims(fImg, 0)

        return fImg