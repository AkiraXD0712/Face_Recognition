# -*- coding: utf-8 -*-
import random
import argparse
import numpy as np
from sklearn.cross_validation import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.models import load_model

from QR_input import extract_data, resize_with_pad, IMAGE_SIZE

FILE_PATH = 'C:\\Users\Akira.DESKTOP-HM7OVCC\.keras\QR_code\model.h5'


class Dataset(object):

    def __init__(self):
        self.X_train = None
        self.X_valid = None
        self.X_test = None
        self.Y_train = None
        self.Y_valid = None
        self.Y_test = None

    def read(self, input_dir, nb_classes, img_rows=IMAGE_SIZE, img_cols=IMAGE_SIZE, img_channels=3):
        images, labels = extract_data(input_dir)
        labels = np.reshape(labels, [-1])
        # numpy.reshape
        X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.3, random_state=random.randint(0, 100))
        X_valid, X_test, y_valid, y_test = train_test_split(images, labels, test_size=0.5, random_state=random.randint(0, 100))

        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, img_channels)
        X_valid = X_valid.reshape(X_valid.shape[0], img_rows, img_cols, img_channels)
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, img_channels)
        input_shape = (img_rows, img_cols, img_channels)

        # the data, shuffled and split between train and test sets
        print('X_train shape:', X_train.shape)
        print(X_train.shape[0], 'train samples')
        print(X_valid.shape[0], 'valid samples')
        print(X_test.shape[0], 'test samples')

        # convert class vectors to binary class matrices
        Y_train = np_utils.to_categorical(y_train, nb_classes)
        Y_valid = np_utils.to_categorical(y_valid, nb_classes)
        Y_test = np_utils.to_categorical(y_test, nb_classes)

        X_train = X_train.astype('float32')
        X_valid = X_valid.astype('float32')
        X_test = X_test.astype('float32')
        X_train /= 255
        X_valid /= 255
        X_test /= 255

        self.X_train = X_train
        self.X_valid = X_valid
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_valid = Y_valid
        self.Y_test = Y_test


class Model(object):

    def __init__(self):
        self.model = None

    def build_model(self, dataset, nb_classes):
        self.model = Sequential()

        self.model.add(Convolution2D(32, (3, 3), border_mode='same', input_shape=dataset.X_train.shape[1:]))
        self.model.add(Activation('relu'))
        self.model.add(Convolution2D(32, (3, 3)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Convolution2D(64, (3, 3), border_mode='same'))
        self.model.add(Activation('relu'))
        self.model.add(Convolution2D(64, (3, 3)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Flatten())
        self.model.add(Dense(512))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(nb_classes))
        self.model.add(Activation('softmax'))

        self.model.summary()

    def train(self, dataset, nb_epoch, batch_size=32, data_augmentation=True):
        # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss='categorical_crossentropy',
                           optimizer='SGD',
                           metrics=['accuracy'])
        if not data_augmentation:
            print('<--------------------Not using data augmentation-------------------->')
            self.model.fit(dataset.X_train, dataset.Y_train,
                           batch_size=batch_size,
                           epochs=nb_epoch,
                           validation_data=(dataset.X_valid, dataset.Y_valid),
                           shuffle=True)
        else:
            print('<--------------------Using real-time data augmentation-------------------->')
            # this will do pre-processing and real-time data augmentation
            datagen = ImageDataGenerator(
                featurewise_center=False,             # set input mean to 0 over the dataset
                samplewise_center=False,              # set each sample mean to 0
                featurewise_std_normalization=False,  # divide inputs by std of the dataset
                samplewise_std_normalization=False,   # divide each input by its std
                zca_whitening=False,                  # apply ZCA whitening
                rotation_range=20,                     # randomly rotate images in the range (degrees, 0 to 180)
                width_shift_range=0.2,                # randomly shift images horizontally (fraction of total width)
                height_shift_range=0.2,               # randomly shift images vertically (fraction of total height)
                horizontal_flip=True,                 # randomly flip images
                vertical_flip=False)                  # randomly flip images

            # compute quantities required for feature wise normalization
            # (std, mean, and principal components if ZCA whitening is applied)
            datagen.fit(dataset.X_train)

            # fit the model on the batches generated by datagen.flow()
            self.model.fit_generator(datagen.flow(dataset.X_train, dataset.Y_train,
                                                  batch_size=batch_size),
                                     samples_per_epoch=dataset.X_train.shape[0],
                                     epochs=nb_epoch,
                                     validation_data=(dataset.X_valid, dataset.Y_valid))

    def save(self, file_path):
        print('<--------------------Model Saved-------------------->')
        self.model.save(file_path)

    def load(self, file_path):
        print('<--------------------Model Loaded-------------------->')
        self.model = load_model(file_path)

    def predict(self, image):
        if image.shape != (1, IMAGE_SIZE, IMAGE_SIZE, 3):
            image = resize_with_pad(image)
            image = image.reshape((1, IMAGE_SIZE, IMAGE_SIZE, 3))
        image = image.astype('float32')
        image /= 255
        result = self.model.predict_proba(image)
        print(result)

        return result

    def evaluate(self, dataset):
        score = self.model.evaluate(dataset.X_test, dataset.Y_test, verbose=0)
        print("%s: %.2f%%" % (self.model.metrics_names[1], score[1] * 100))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, help='path to read input')
    parser.add_argument("--model_dir", type=str, help='path to save model')
    args = parser.parse_args()

    if args.input_dir:
        dataset = Dataset()
        dataset.read(input_dir=args.input_dir, nb_classes=2)

        model = Model()
        model.build_model(dataset, nb_classes=2)
        model.train(dataset, nb_epoch=10, batch_size=32, data_augmentation=True)
        if args.model_dir:
            model.save(file_path=args.input_dir)

            model = Model()
            model.load(file_path=args.model_dir)
            model.evaluate(dataset)
        else:
            print("<--------------------Model no saved-------------------->")
            pass
    else:
        print("<--------------------Input no found-------------------->")
        pass