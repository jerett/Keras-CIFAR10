from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.layers import BatchNormalization
from keras.layers import MaxPool2D


def simple_cnn(classes, input_shape):
    """
    * 7x7 layer conv with 32 filters, stride is (2,2)
    * ReLU
    * Dense Layer
    * SVM
    """
    model = Sequential()
    model.add(Conv2D(filters=32,
                            kernel_size=7,
                            # kernel_regularizer=keras.regularizers.l2,
                            strides=2,
                            input_shape=input_shape,
                            ))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(classes))
    model.add(Activation('linear'))
    return model


def complex_cnn(classes, input_shape):
    """
    * 7x7 layer Conv layer with 32 filters
    * BN Layer
    * ReLu Activation Layer
    * 2x2 Max Polling Layer
    * Dense Layer with 1024 output units.
    * BN Layer
    * ReLU
    * Dense Layer with 10 outputs.
    """
    complex_model = Sequential()
    complex_model.add(Conv2D(filters=32,
                             kernel_size=7,
                             padding='same',
                             input_shape=input_shape,
                             ))
    complex_model.add(BatchNormalization())
    complex_model.add(Activation('relu'))
    complex_model.add(MaxPool2D(padding='same'))
    complex_model.add(Flatten())
    complex_model.add(Dense(1024))
    complex_model.add(BatchNormalization())
    complex_model.add(Activation('relu'))
    complex_model.add(Dense(classes))
    complex_model.add(Activation('softmax'))
    return complex_model
