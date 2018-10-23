import h5py
import keras.applications
import cv2
import numpy as np
import os
import json
from keras.preprocessing import image
from keras.callbacks import ModelCheckpoint
from keras.models import Model
from keras.layers import Input, Dense, Flatten
from keras.utils import to_categorical

class TransferModel():

    def __init__(self, final_layer_weight, class_labels_path, img_shape = (224, 224), tranfer_model_name = "inception"):

        model, preprocess_input = get_features_CNN(tranfer_model_name, [3] + list(img_shape))
        self.transfer_model = model
        self.preprocess_input = preprocess_input
        self.img_shape = img_shape

        self.class_labels = class_labels(class_labels_path)
        self.final_model = get_softmax_model(model.output_shape[1:], len(self.class_labels))
        self.final_model.load_weights(final_layer_weight)

    def run(self, input_x):
        input_x = self.preprocess_input(input_x)

        features = self.transfer_model.predict(input_x)
        result = self.final_model.predict(features)

        return result

    def load_images(self, file_paths):
        imgs = []
        for img_file in file_paths:
            img = image.load_img(img_file, target_size=self.img_shape)
            imgs.append(image.img_to_array(img))
        return np.stack(imgs, axis=0)

    def predict_ims(self, imgs, resize=False, reverse_channel=False):
        img_arr = []
        if resize:
            for im in imgs:
                im = cv2.resize(im, self.img_shape)
                if reverse_channel:
                    im = im[:,:,::-1]
                img_arr.append(image.img_to_array(im))
        img_arr = np.stack(img_arr, axis=0)
        print("Predicting...")
        y_proba = self.run(img_arr)
        y_ind = np.argmax(y_proba, axis=-1)
        labels = [self.class_labels[str(y)] for y in y_ind]

        return labels

    def predict_ims_from_path(self, file_paths):
        print("Loading ims")
        input_x = self.load_images(file_paths)

        return self.predict_ims(input_x)

def get_features_CNN(model_name, img_shape):

    if model_name == "vgg19":
        model = keras.applications.vgg19.VGG19(weights="imagenet", include_top=False,
                                               input_shape=img_shape)
        preprocess_input = keras.applications.vgg19.preprocess_input
    elif model_name == "vgg16":
        model = keras.applications.vgg16.VGG16(weights="imagenet", include_top=False,
                                               input_shape=img_shape)
        preprocess_input = keras.applications.vgg16.preprocess_input
    elif model_name == "inception":
        model = keras.applications.inception_v3.InceptionV3(weights="imagenet",
                                                            include_top=False, input_shape=img_shape)
        preprocess_input = keras.applications.inception_v3.preprocess_input
    else:
        model, preprocess_input = None, None

    return model, preprocess_input


def class_labels(label_json_path):
    return json.load(open(label_json_path, "r"))

def get_softmax_model(input_shape, num_classes):
    inputs = Input(input_shape)
    flatten = Flatten(name="flatten")(inputs)
    predictions = Dense(num_classes, activation="softmax")(flatten)

    model = Model(inputs, predictions, name="softmax-classification")

    return model

if __name__ == "__main__":
    from os import listdir
    from os.path import isfile, join

    input_folder = "/home/taprosoft/Downloads/test_segmented/flax_bprost/bprost/demo_images/license_cells/color/samples/one_sample/大型"
    file_paths = [join(input_folder, f) for f in listdir(input_folder) if isfile(join(input_folder, f))]

    model = TransferModel("inception", "data/inception_None_weights.02-1.00.hdf5", "data/labels.json", [224, 224])
    labels = model.predict_ims_from_path(file_paths)

    for f, l in zip(file_paths, labels):
        print(f, l)
