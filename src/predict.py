# -*- coding: utf-8 -*-
from train import Model
from input import read_image
import argparse
import os
import numpy as np


def prediction(path, classes):
    for file in os.listdir(path):
        if file.endswith('.jpg') or file.endswith('.png'):
            print('test %s' % file)
            file_path = os.path.abspath(os.path.join(path, file))
            image = read_image(file_path)
            result = model.predict(image)
            index = np.argmax(result)
            print(classes[index], result[index])

if __name__ == '__main__':
    classes = []
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--predict_dir',
        type=str,
        help='folder of images'
    )
    args = parser.parse_args()
    if args.predict_dir:
        model = Model()
        try:
            model.load(file_path=args.predict_dir + '\model.h5')
            with open(args.predict_dir + '\labels.txt', 'r') as f:
                for line in f.readlines():
                    classes.append(line.strip())
        except OSError as e:
            print("<--------------------Unable to open file-------------------->\n", e)
        else:
            prediction(args.predict_dir, classes)
    else:
        print('Input no found\nTry "python predict.py -h" for more information')



