# -*- coding: utf-8 -*-
import cv2
import argparse
import time
import numpy as np
from training import Model

classes = []
FRAME_SIZE = 256
font = cv2.FONT_HERSHEY_SIMPLEX
switch = False


def detect(image):
    crop_image = image[112:112 + FRAME_SIZE, 192:192 + FRAME_SIZE]
    result = model.predict(crop_image)
    index = np.argmax(result)
    cv2.putText(image, classes[index], (192, 112), font, 1, (0, 255, 0), 2)


def crop_save(image):
    crop_image = image[112 + 2:112 + FRAME_SIZE - 2, 192 + 2:192 + FRAME_SIZE - 2]
    timestamp = str(time.time())
    cv2.imwrite(
        'C:\\Users\Akira.DESKTOP-HM7OVCC\Desktop\database\\' + timestamp + '.png',
        crop_image,
        (cv2.IMWRITE_PNG_COMPRESSION, 0)
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_dir',
        type=str,
        help='folder contains model and labels'
    )
    args = parser.parse_args()

    if args.model_dir:
        model = Model()
        try:
            model.load(file_path=args.model_dir + '\model.h5')
            with open(args.model_dir + '\labels.txt', 'r') as f:
                for line in f.readlines():
                    classes.append(line.strip())
        except OSError as e:
            print("<--------------------Unable to open file-------------------->\n", e)
        else:
            cv2.namedWindow('Video')

            # open le camera
            capture = cv2.VideoCapture(0)

            while capture.isOpened():

                _, frame = capture.read()
                cv2.rectangle(frame, (192, 112), (192 + FRAME_SIZE, 112 + FRAME_SIZE), (0, 255, 0), 2)
                if switch:
                    detect(frame)
                cv2.imshow('Video', frame)
                key = cv2.waitKey(10)
                if key == ord('z'):
                    switch = True
                elif key == ord('d'):
                    switch = False
                elif key == ord('s'):
                    crop_save(frame)
                elif key == ord('q'):  # exit
                    break

            capture.release()
            cv2.destroyWindow('Video')
    else:
        print('Input no found\nTry "python predict.py -h" for more information')



