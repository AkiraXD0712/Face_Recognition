from QR_train import Model, FILE_PATH
import cv2

model = Model()
model.load(file_path=FILE_PATH)
for i in range(1, 5):
    image = cv2.imread('C:\\Users\Akira.DESKTOP-HM7OVCC\Desktop\\test\\test_' + str(i) + '.jpg')
    result = model.predict(image)
    if result[0][0] >= 0.99:
        print('this is QR_code')
    else:
        print('This is not QR_code')

image = cv2.imread('C:\\Users\Akira.DESKTOP-HM7OVCC\Desktop\\test\\test_' + str(5) + '.png')
result = model.predict(image)
if result[0][0] >= 0.99:  # boss
    print('this is QR_code')
else:
    print('This is not QR_code')

image = cv2.imread('C:\\Users\Akira.DESKTOP-HM7OVCC\Desktop\\test\\test_' + str(6) + '.jpg')
result = model.predict(image)
if result[0][0] >= 0.99:  # boss
    print('this is QR_code')
else:
    print('This is not QR_code')
