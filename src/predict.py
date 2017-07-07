from QR_train import Model, MODEL_DIR
from QR_input import read_image, CLASSES
import argparse
import os
import numpy as np


def prediction(path):
    for file in os.listdir(path):
        print('test %s' % file)
        file_path = os.path.abspath(os.path.join(path, file))
        image = read_image(file_path)
        result = model.predict(image)
        index = np.argmax(result)
        print("This is %s: %2.f" % (CLASSES[index], result[index]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--predict_dir',
        type=str,
        help='folder of images'
    )
    args = parser.parse_args()
    if args.predict_dir:
        model = Model()
        model.load(file_path=MODEL_DIR)
        prediction(args.predict_dir)
    else:
        print('Input no found\nTry "python predict.py -h" for more information')




