# detect face
# Author: CHEN Lichong
# Date: 05/04/2017

import cv2
from keras.models import model_from_json
import numpy as np

font = cv2.FONT_HERSHEY_SIMPLEX

face_cascade = cv2.CascadeClassifier("C:\\Users\Akira\Desktop\haarcascade_frontalface_default.xml")

num = 0
path = 'C:\\Users\Akira\Desktop\photo\\Chen Lichong'
# position face detected
position_x = 0
position_y = 0
width = 0
high = 0


def detect_face(img):
    global position_x, position_y, width, high
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face = face_cascade.detectMultiScale(
        image=gray,
        scaleFactor=1.2,
        minNeighbors=2,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    for (x, y, w, h) in face:
        cv2.rectangle(img, (position_x, position_y), (position_x + width, position_y + high), (0, 255, 0), 2)
        position_x = x
        position_y = y
        width = w
        high = h


def crop_face(img):
    global position_x, position_y, width, high
    img = img[position_y-20:position_y + high+20, position_x:position_x + width]
    crop_img = cv2.resize(img, (200, 200), interpolation=cv2.INTER_LINEAR)
    return crop_img


# load json and create model
json_file = open('C:\\Users\Akira\.keras\deeplearning\model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("C:\\Users\Akira\.keras\deeplearning\deep_learning.h5")
print("Loaded model from disk")

cv2.namedWindow('Video')

capture = cv2.VideoCapture(0)

while capture.isOpened():

    _, frame = capture.read()
    frame_copy = frame.copy()
    detect_face(frame_copy)

    visage = crop_face(frame)
    visage = cv2.cvtColor(visage, cv2.COLOR_BGR2GRAY)

    cc = np.array(visage)
    cc = np.array(cc)
    cc = cc.reshape(1, 200, 200, 1)
    cc = cc.astype('float32')
    cc /= 255

    predictions = loaded_model.predict(cc)

    if predictions[0][0] > 0.95:
        cv2.putText(frame_copy, 'this is jennifer', (position_x, position_y + high), font, 1, (0, 255, 0), 2)
    elif predictions[0][1] > 0.95:
        cv2.putText(frame_copy, 'this is lichong', (position_x, position_y + high), font, 1, (0, 255, 0), 2)
    elif predictions[0][2] > 0.95:
        cv2.putText(frame_copy, 'this is tongtong', (position_x, position_y + high), font, 1, (0, 255, 0), 2)
    elif predictions[0][3] > 0.95:
        cv2.putText(frame_copy, 'this is walia', (position_x, position_y + high), font, 1, (0, 255, 0), 2)
    else:
        cv2.putText(frame_copy, "can't recognized", (position_x, position_y + high), font, 1, (0, 255, 0), 2)

    cv2.imshow('Video', frame_copy)

    key = cv2.waitKey(20)
    if key == ord('q'):  # exit
        break

capture.release()
cv2.destroyWindow('Video')