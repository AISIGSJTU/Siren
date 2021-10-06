#########################
# Purpose: Model definitions and other utlities for MNIST, Fashion MNIST, 
########################

import tensorflow as tf
from keras.datasets import mnist
from keras.callbacks import EarlyStopping
from keras.models import Sequential, model_from_json, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, MaxPool2D, GlobalAveragePooling2D, Layer, Add
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
import global_vars as gv

from utils.fmnist import load_fmnist
import global_vars as gv

import argparse
import numpy as np
# np.random.seed(777)

def data_mnist(one_hot=True):
    """
    Preprocess MNIST dataset
    """
    # the data, shuffled and split between train and test sets
    if gv.args.dataset == 'MNIST':
        (X_train, y_train), (X_test, y_test) = mnist.load_data()

    elif gv.args.dataset == 'fMNIST':
        X_train, y_train = load_fmnist('/data/hanxiguo/data/', kind='train')
        X_test, y_test = load_fmnist('/data/hanxiguo/data/', kind='t10k')


    X_train = X_train.reshape(X_train.shape[0],
                              gv.IMAGE_ROWS,
                              gv.IMAGE_COLS,
                              gv.NUM_CHANNELS)

    X_test = X_test.reshape(X_test.shape[0],
                            gv.IMAGE_ROWS,
                            gv.IMAGE_COLS,
                            gv.NUM_CHANNELS)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    if gv.args.gar == 'siren':
        target_indices = np.random.choice(len(X_test), gv.args.root_size)
        Server_X = X_train[target_indices]
        Server_Y = y_train[target_indices]
        print("server dataset initialized..")
        print('server dataset shape:', Server_X.shape)
        X_train = np.delete(X_train, target_indices, axis=0)
        y_train = np.delete(y_train, target_indices, axis=0)
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    if one_hot:
        # convert class vectors to binary class matrices
        y_train = np_utils.to_categorical(y_train, gv.NUM_CLASSES).astype(np.float32)
        y_test = np_utils.to_categorical(y_test, gv.NUM_CLASSES).astype(np.float32)
    if gv.args.gar == 'siren':
        return X_train, y_train, X_test, y_test, Server_X, Server_Y
    else:
        return X_train, y_train, X_test, y_test



def model_cifar():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(10))
    return model

def modelA():
    model = Sequential()
    model.add(Conv2D(64, (5, 5), padding='valid', input_shape=(gv.IMAGE_ROWS,
                                         gv.IMAGE_COLS,
                                         gv.NUM_CHANNELS)))
    model.add(Activation('relu'))

    model.add(Conv2D(64, (5, 5)))
    model.add(Activation('relu'))

    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))

    model.add(Dropout(0.5))
    model.add(Dense(gv.NUM_CLASSES))
    return model


def modelB():
    model = Sequential()
    model.add(Dropout(0.2, input_shape=(gv.IMAGE_ROWS,
                                        gv.IMAGE_COLS,
                                        gv.NUM_CHANNELS)))
    model.add(Convolution2D(64, 8, 8,
                            subsample=(2, 2),
                            border_mode='same'))
    model.add(Activation('relu'))

    model.add(Convolution2D(128, 6, 6,
                            subsample=(2, 2),
                            border_mode='valid'))
    model.add(Activation('relu'))

    model.add(Convolution2D(128, 5, 5,
                            subsample=(1, 1)))
    model.add(Activation('relu'))

    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(gv.NUM_CLASSES))
    return model


def modelC():
    model = Sequential()
    model.add(Convolution2D(128, 3, 3,
                            border_mode='valid',
                            input_shape=(gv.IMAGE_ROWS,
                                         gv.IMAGE_COLS,
                                         gv.NUM_CHANNELS)))
    model.add(Activation('relu'))

    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))

    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))

    model.add(Dropout(0.5))
    model.add(Dense(gv.NUM_CLASSES))
    return model


def modelD():
    model = Sequential()

    model.add(Flatten(input_shape=(gv.IMAGE_ROWS,
                                   gv.IMAGE_COLS,
                                   gv.NUM_CHANNELS)))

    model.add(Dense(300, init='he_normal', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(300, init='he_normal', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(300, init='he_normal', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(300, init='he_normal', activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(gv.NUM_CLASSES))
    return model

def modelE():
    model = Sequential()

    model.add(Flatten(input_shape=(gv.IMAGE_ROWS,
                                   gv.IMAGE_COLS,
                                   gv.NUM_CHANNELS)))

    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))

    model.add(Dense(gv.NUM_CLASSES))

    return model

def modelF():
    model = Sequential()

    model.add(Conv2D(32, (5, 5),
                            padding='valid',
                            input_shape=(gv.IMAGE_ROWS,
                                         gv.IMAGE_COLS,
                                         gv.NUM_CHANNELS)))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (5, 5)))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))

    model.add(Dense(gv.NUM_CLASSES))

    return model

def modelG():
    model = Sequential()

    model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                     activation ='relu', input_shape = (28,28,1)))
    model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                     activation ='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))


    model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                     activation ='relu'))
    model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                     activation ='relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    model.add(Dropout(0.25))


    model.add(Flatten())
    model.add(Dense(512, activation = "relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(10))

    return model

def model_LR():
    model = Sequential()

    model.add(Flatten(input_shape=(gv.IMAGE_ROWS,
                                   gv.IMAGE_COLS,
                                   gv.NUM_CHANNELS)))

    model.add(Dense(gv.NUM_CLASSES))

    return model

class ResnetBlock(Model):
    """
    A standard resnet block.
    """

    def __init__(self, channels: int, down_sample=False):
        """
        channels: same as number of convolution kernels
        """
        super().__init__()

        self.__channels = channels
        self.__down_sample = down_sample
        self.__strides = [2, 1] if down_sample else [1, 1]

        KERNEL_SIZE = (3, 3)
        # use He initialization, instead of Xavier (a.k.a 'glorot_uniform' in Keras), as suggested in [2]
        INIT_SCHEME = "he_normal"

        self.conv_1 = Conv2D(self.__channels, strides=self.__strides[0],
                             kernel_size=KERNEL_SIZE, padding="same", kernel_initializer=INIT_SCHEME)
        self.bn_1 = BatchNormalization()
        self.conv_2 = Conv2D(self.__channels, strides=self.__strides[1],
                             kernel_size=KERNEL_SIZE, padding="same", kernel_initializer=INIT_SCHEME)
        self.bn_2 = BatchNormalization()
        self.merge = Add()

        if self.__down_sample:
            # perform down sampling using stride of 2, according to [1].
            self.res_conv = Conv2D(
                self.__channels, strides=2, kernel_size=(1, 1), kernel_initializer=INIT_SCHEME, padding="same")
            self.res_bn = BatchNormalization()

    def call(self, inputs):
        res = inputs

        x = self.conv_1(inputs)
        x = self.bn_1(x)
        x = tf.nn.relu(x)
        x = self.conv_2(x)
        x = self.bn_2(x)

        if self.__down_sample:
            res = self.res_conv(res)
            res = self.res_bn(res)

        # if not perform down sample, then add a shortcut directly
        x = self.merge([x, res])
        out = tf.nn.relu(x)
        return out


class ResNet18(Model):

    def __init__(self, num_classes, **kwargs):
        """
            num_classes: number of classes in specific classification task.
        """
        super().__init__(**kwargs)
        self.conv_1 = Conv2D(64, (7, 7), strides=2,
                             padding="same", kernel_initializer="he_normal")
        self.init_bn = BatchNormalization()
        self.pool_2 = MaxPool2D(pool_size=(2, 2), strides=2, padding="same")
        self.res_1_1 = ResnetBlock(64)
        self.res_1_2 = ResnetBlock(64)
        self.res_2_1 = ResnetBlock(128, down_sample=True)
        self.res_2_2 = ResnetBlock(128)
        self.res_3_1 = ResnetBlock(256, down_sample=True)
        self.res_3_2 = ResnetBlock(256)
        self.res_4_1 = ResnetBlock(512, down_sample=True)
        self.res_4_2 = ResnetBlock(512)
        self.avg_pool = GlobalAveragePooling2D()
        self.flat = Flatten()
        self.fc = Dense(num_classes, activation="softmax")

    def call(self, inputs):
        out = self.conv_1(inputs)
        out = self.init_bn(out)
        out = tf.nn.relu(out)
        out = self.pool_2(out)
        for res_block in [self.res_1_1, self.res_1_2, self.res_2_1, self.res_2_2, self.res_3_1, self.res_3_2, self.res_4_1, self.res_4_2]:
            out = res_block(out)
        out = self.avg_pool(out)
        out = self.flat(out)
        out = self.fc(out)
        return out

def model_resnet18():
    model = ResNet18(gv.NUM_CLASSES)
    return model


def model_mnist(type):
    """
    Defines MNIST model using Keras sequential model
    """

    models = [model_cifar, modelA, modelB, modelC, modelD, modelE, modelF, modelG, model_LR, model_resnet18]

    return models[type]()


def data_gen_mnist(X_train):
    datagen = ImageDataGenerator()

    datagen.fit(X_train)
    return datagen


def load_model(model_path, type=0):

    try:
        with open(model_path+'.json', 'r') as f:
            json_string = f.read()
            model = model_from_json(json_string)
            print('Loaded using json')
    except IOError:
        model = model_mnist(type=type)

    model.load_weights(model_path)
    return model
