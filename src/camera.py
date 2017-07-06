# detect face
# Author: CHEN Lichong
# Date: 05/04/2017

import cv2

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
    crop_img = cv2.resize(img[position_y-20:position_y + high+20, position_x:position_x + width], (400, 400))
    return crop_img


cv2.namedWindow('Video')

capture = cv2.VideoCapture(0)

while capture.isOpened():

    _, frame = capture.read()
    frame_copy = frame.copy()
    detect_face(frame_copy)

    cv2.imshow('Video', frame_copy)

    key = cv2.waitKey(20)
    if key == ord('s'):     # capture
        visage = crop_face(frame)
        filename = path + str(num) + '.png'
        cv2.imwrite(filename, visage, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
        num += 1
    elif key == ord('q'): # exit
        break

capture.release()
cv2.destroyWindow('Video')
