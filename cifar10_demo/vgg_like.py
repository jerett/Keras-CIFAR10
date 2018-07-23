import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, Conv2D, MaxPool2D, BatchNormalization, Activation


# from sklearn.manifold import TSNE
# from sklearn import manifold
# from sklearn import cluster
# from sklearn.preprocessing import StandardScaler

def vgg16_like(classes, input_shape=(224, 224, 3), scale_factor=1, weights=None, weight_decay=.0):
    f = scale_factor
    model = Sequential()
    # model.add(ZeroPadding2D((1, 1), input_shape=input_shape))
    # model.add(Convolution2D(32 // f, 3, 3, activation='relu'))
    # model.add(ZeroPadding2D((1, 1)))
    # model.add(Convolution2D(32 // f, 3, 3, activation='relu'))
    # model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # block 1
    # model.add(Conv2D(filters=32 // f, kernel_size=(3, 3), padding='same', activation='relu'))
    # model.add(Conv2D(filters=32 // f, kernel_size=(3, 3), padding='same', activation='relu'))

    model.add(Conv2D(filters=64 // f,
                     kernel_size=(3, 3),
                     padding='same',
                     kernel_regularizer=keras.regularizers.l2(weight_decay),
                     ))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(filters=64 // f,
                     kernel_size=(3, 3),
                     padding='same',
                     kernel_regularizer=keras.regularizers.l2(weight_decay),
                     ))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPool2D())
    model.add(Dropout(0.3))

    # model.add(ZeroPadding2D((1, 1)))
    # model.add(Convolution2D(64 // f, 3, 3, activation='relu'))
    # model.add(ZeroPadding2D((1, 1)))
    # model.add(Convolution2D(64 // f, 3, 3, activation='relu'))
    # model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    # block 2
    # model.add(Conv2D(filters=64 // f, kernel_size=(3, 3), padding='same', activation='relu'))
    # model.add(Conv2D(filters=64 // f, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(Conv2D(filters=128 // f,
                     kernel_size=(3, 3),
                     padding='same',
                     kernel_regularizer=keras.regularizers.l2(weight_decay),
                     ))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(filters=128 // f,
                     kernel_size=(3, 3),
                     padding='same',
                     kernel_regularizer=keras.regularizers.l2(weight_decay),
                     ))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPool2D())
    model.add(Dropout(0.4))

    # model.add(ZeroPadding2D((1, 1)))
    # model.add(Convolution2D(128 // f, 3, 3, activation='relu'))
    # model.add(ZeroPadding2D((1, 1)))
    # model.add(Convolution2D(128 // f, 3, 3, activation='relu'))
    # model.add(ZeroPadding2D((1, 1)))
    # model.add(Convolution2D(128 // f, 3, 3, activation='relu'))
    # model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    # block 3
    # model.add(Conv2D(filters=128 // f, kernel_size=(3, 3), padding='same', activation='relu'))
    # model.add(Conv2D(filters=128 // f, kernel_size=(3, 3), padding='same', activation='relu'))
    # model.add(Conv2D(filters=128 // f, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(Conv2D(filters=256 // f,
                     kernel_size=(3, 3),
                     padding='same',
                     kernel_regularizer=keras.regularizers.l2(weight_decay),))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(filters=256 // f,
                     kernel_size=(3, 3),
                     padding='same',
                     kernel_regularizer=keras.regularizers.l2(weight_decay),))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(filters=256 // f,
                     kernel_size=(3, 3),
                     padding='same',
                     kernel_regularizer=keras.regularizers.l2(weight_decay),))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPool2D())
    model.add(Dropout(0.4))

    # model.add(ZeroPadding2D((1, 1)))
    # model.add(Convolution2D(256 // f, 3, 3, activation='relu'))
    # model.add(ZeroPadding2D((1, 1)))
    # model.add(Convolution2D(256 // f, 3, 3, activation='relu'))
    # model.add(ZeroPadding2D((1, 1)))
    # model.add(Convolution2D(256 // f, 3, 3, activation='relu'))
    # model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    # block 4
    # model.add(Conv2D(filters=256 // f, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(Conv2D(filters=512 // f,
                     kernel_size=(3, 3),
                     padding='same',
                     kernel_regularizer=keras.regularizers.l2(weight_decay),))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    # model.add(Conv2D(filters=256 // f, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(Conv2D(filters=512 // f,
                     kernel_size=(3, 3),
                     padding='same',
                     kernel_regularizer=keras.regularizers.l2(weight_decay),))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    # model.add(Conv2D(filters=256 // f, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(Conv2D(filters=512 // f,
                     kernel_size=(3, 3),
                     padding='same',
                     kernel_regularizer=keras.regularizers.l2(weight_decay),))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPool2D())
    model.add(Dropout(0.4))

    # model.add(ZeroPadding2D((1, 1)))
    # model.add(Convolution2D(256 // f, 3, 3, activation='relu'))
    # model.add(ZeroPadding2D((1, 1)))
    # model.add(Convolution2D(256 // f, 3, 3, activation='relu'))
    # model.add(ZeroPadding2D((1, 1)))
    # model.add(Convolution2D(256 // f, 3, 3, activation='relu'))
    # model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    # block 5
    # model.add(Conv2D(filters=256 // f, kernel_size=(3, 3), padding='same', activation='relu'))
    # model.add(Conv2D(filters=256 // f, kernel_size=(3, 3), padding='same', activation='relu'))
    # model.add(Conv2D(filters=256 // f, kernel_size=(3, 3), padding='same', activation='relu'))
    # model.add(MaxPool2D())
    # model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(classes, activation='softmax', name='predictions'))
    return model
