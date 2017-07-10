# -*- coding: utf-8 -*-
import os
from sklearn import preprocessing
import numpy as np
import cv2

IMAGE_SIZE = 64
images = []
labels = []


def resize_with_pad(image, height=IMAGE_SIZE, width=IMAGE_SIZE):

    def get_padding_size(image):
        h, w, _ = image.shape
        longest_edge = max(h, w)
        top, bottom, left, right = (0, 0, 0, 0)
        if h < longest_edge:
            dh = longest_edge - h
            top = dh // 2
            bottom = dh - top
        elif w < longest_edge:
            dw = longest_edge - w
            left = dw // 2
            right = dw - left
        else:
            pass
        return top, bottom, left, right

    top, bottom, left, right = get_padding_size(image)
    black = [0, 0, 0]
    constant = cv2.copyMakeBorder(image, top , bottom, left, right, cv2.BORDER_CONSTANT, value=black)
    resized_image = cv2.resize(constant, (height, width))

    return resized_image


def traverse_dir(path):
    for file_or_dir in os.listdir(path):
        abs_path = os.path.abspath(os.path.join(path, file_or_dir))
        # dir
        if os.path.isdir(abs_path):
            traverse_dir(abs_path)
        # file
        else:
            if file_or_dir.endswith('.jpg') or file_or_dir.endswith('.png'):
                image = read_image(abs_path)
                images.append(image)

                dir_name, _ = os.path.split(abs_path)
                label = os.path.basename(dir_name)
                labels.append(label)
    return images, labels


def read_image(file_path):
    image = cv2.imread(file_path)
    image = resize_with_pad(image, IMAGE_SIZE, IMAGE_SIZE)

    return image


def extract_data(path):
    global CLASSES
    images, labels = traverse_dir(path)
    images = np.array(images)
    # change to ont-hot vector
    one_hot = preprocessing.LabelBinarizer()
    one_hot.fit(labels)
    nb_classes = len(one_hot.classes_)

    with open(path+'\labels.txt', 'w') as f:
        for label in one_hot.classes_:
            f.write(label + '\n')

    one_hots = list(one_hot.transform([i]) for i in labels)
    one_hots = np.array(one_hots)
    one_hots = np.reshape(one_hots, (images.shape[0], nb_classes))

    return images, one_hots, nb_classes


if __name__ == '__main__':
    images, labels, nb_classes = extract_data('C:\\Users\Akira.DESKTOP-HM7OVCC\Desktop\photo')
    print(images.shape, labels.shape)
    print(labels)

