# -*- coding: utf-8 -*-
import random
import argparse
from sklearn.cross_validation import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.models import load_model
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform, conditional
from input import extract_data, resize_with_pad, IMAGE_SIZE


class Dataset(object):

    def __init__(self):
        self.x_train = None
        self.x_valid = None
        self.x_test = None
        self.y_train = None
        self.y_valid = None
        self.y_test = None

    def read(self, input_dir):
        images, labels, nb_classes = extract_data(input_dir)

        # shuffle and split data between train and test sets
        x_train, x_test, y_train, y_test = train_test_split(
            images,
            labels,
            test_size=0.3,
            random_state=random.randint(0, 100)
        )
        x_valid, x_test, y_valid, y_test = train_test_split(
            images,
            labels,
            test_size=0.5,
            random_state=random.randint(0, 100)
        )

        print('x_train shape:', x_train.shape)
        print(x_train.shape[0], 'train samples')
        print(x_valid.shape[0], 'valid samples')
        print(x_test.shape[0], 'test samples')

        # # convert class vectors to binary class matrices
        # y_train = np_utils.to_categorical(y_train, nb_classes)
        # y_valid = np_utils.to_categorical(y_valid, nb_classes)
        # y_test = np_utils.to_categorical(y_test, nb_classes)

        x_train = x_train.astype('float32')
        x_valid = x_valid.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_valid /= 255
        x_test /= 255

        self.x_train = x_train
        self.x_valid = x_valid
        self.x_test = x_test
        self.y_train = y_train
        self.y_valid = y_valid
        self.y_test = y_test

        return nb_classes


class Model(object):

    def __init__(self):
        self.model = None

    def build_model(self, dataset, nb_classes):
        self.model = Sequential()

        self.model.add(Convolution2D(32, (3, 3), padding='same', input_shape=dataset.x_train.shape[1:]))
        self.model.add(Activation('relu'))
        self.model.add(Convolution2D(32, (3, 3)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Convolution2D(64, (3, 3), padding='same'))
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

    def train(self, dataset, nb_epoch, batch_size, data_augmentation):
        # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss='categorical_crossentropy',
                           optimizer='SGD',
                           metrics=['accuracy'])
        if not data_augmentation:
            print('<--------------------Not using data augmentation-------------------->')
            self.model.fit(dataset.x_train, dataset.y_train,
                           batch_size=batch_size,
                           epochs=nb_epoch,
                           validation_data=(dataset.x_valid, dataset.y_valid),
                           shuffle=True)
        else:
            print('<--------------------Using real-time data augmentation-------------------->')
            # do pre-processing and real-time data augmentation
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
            datagen.fit(dataset.x_train)

            # fit the model on the batches generated by datagen.flow()
            self.model.fit_generator(datagen.flow(dataset.x_train, dataset.y_train, batch_size=batch_size),
                                     steps_per_epoch=dataset.x_train.shape[0],
                                     epochs=nb_epoch,
                                     validation_data=(dataset.x_valid, dataset.y_valid))

    def save(self, file_path):
        print('<--------------------Model Saved at %s-------------------->' % file_path)
        self.model.save(file_path)

    def load(self, file_path):
        print('<--------------------Model Loaded from %s-------------------->' % file_path)
        self.model = load_model(file_path)

    def predict(self, image):
        if image.shape != (1, IMAGE_SIZE, IMAGE_SIZE, 3):
            image = resize_with_pad(image)
            image = image.reshape((1, IMAGE_SIZE, IMAGE_SIZE, 3))
        image = image.astype('float32')
        image /= 255
        result = self.model.predict_proba(image)
        return result[0]

    def evaluate(self, dataset):
        score = self.model.evaluate(dataset.x_test, dataset.y_test, verbose=0)
        print("%s: %.2f%%" % (self.model.metrics_names[1], score[1] * 100))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_dir',
        type=str,
        help='path to read input'
    )
    parser.add_argument(
        '--epoch',
        type=int,
        default=20,
        help='choose number of epoch'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='choose batch size'
    )
    parser.add_argument(
        '--data_augmentation',
        type=bool,
        default=True,
        help='Use real time data augmentation'
    )
    args = parser.parse_args()
    if args.input_dir:
        dataset = Dataset()
        nb_classes = dataset.read(input_dir=args.input_dir)

        model = Model()
        model.build_model(dataset, nb_classes=nb_classes)
        model.train(
            dataset,
            nb_epoch=args.epoch,
            batch_size=args.batch_size,
            data_augmentation=args.data_augmentation
        )
        model.evaluate(dataset)
        model.save(file_path=args.input_dir + '\model.h5')
    else:
        print('Input no found\nTry "python train.py -h" for more information')
        pass
