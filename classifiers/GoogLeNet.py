from keras.models import Model
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Conv2D
from keras.layers import MaxPool2D
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import Input
from keras.layers import AveragePooling2D
from keras import regularizers
from keras.layers import concatenate


def conv2d_bn_relu(x, filters, kernel_size, name, weight_decay=.0, strides=(1, 1), use_bn=True):
    conv_name = name + "-conv"
    x = Conv2D(filters=filters,
               kernel_size=kernel_size,
               strides=strides,
               padding='same',
               use_bias=False,
               kernel_regularizer=regularizers.l2(weight_decay),
               name=conv_name)(x)
    if use_bn:
        bn_name = name + "-bn"
        x = BatchNormalization(scale=False, axis=3, name=bn_name)(x)
    relu_name = name + "-relu"
    x = Activation('relu', name=relu_name)(x)
    return x


def inception_block_v1(x, filters_num_array, name, weight_decay=.0, use_bn=True):
    """

    :param x: model
    :param filters_num_array: filters num is 4 branch format (1x1, (1x1, 3x3), (1x1, 5x5), (pool, 1x1))
    :return: block added model x
    """
    (br0_filters, br1_filters, br2_filters, br3_filters) = filters_num_array
    # br0
    br0 = conv2d_bn_relu(x,
                         filters=br0_filters, kernel_size=(1, 1), weight_decay=weight_decay,
                         name=name + '-br0-1x1', use_bn=use_bn)

    # br1
    br1 = conv2d_bn_relu(x,
                         filters=br1_filters[0], kernel_size=(1, 1), weight_decay=weight_decay,
                         name=name + '-br1-1x1', use_bn=use_bn)
    br1 = conv2d_bn_relu(br1,
                         filters=br1_filters[1], kernel_size=(3, 3), weight_decay=weight_decay,
                         name=name + '-br1-3x3', use_bn=use_bn)

    # br2
    br2 = conv2d_bn_relu(x,
                         filters=br2_filters[0], kernel_size=(1, 1), weight_decay=weight_decay,
                         name=name + '-br2-1x1', use_bn=use_bn)
    br2 = conv2d_bn_relu(br2, filters=br2_filters[1], kernel_size=(5, 5), name=name + '-br2-5x5', use_bn=use_bn)

    # br3
    br3 = MaxPool2D(pool_size=(3, 3), strides=(1, 1), padding='same', name=name + '-br3-pool')(x)
    br3 = conv2d_bn_relu(br3, filters=br3_filters, kernel_size=(1, 1), weight_decay=weight_decay, name=name + '-br3-1x1')

    x = concatenate([br0, br1, br2, br3], axis=3, name=name)
    return x


def aux_classifier_v1(x, num_classes, name):
    aux_classifier = AveragePooling2D(pool_size=(5, 5), strides=(3, 3), name=name + '-averagePool')(x)
    aux_classifier = conv2d_bn_relu(aux_classifier, filters=128, kernel_size=(1, 1), name=name+'-1x1conv')
    aux_classifier = Flatten(name=name + '-flatten')(aux_classifier)
    aux_classifier = Dense(1024)(aux_classifier)
    aux_classifier = Dropout(0.3, name=name + '-dropout')(aux_classifier)
    aux_classifier = Dense(num_classes, activation='softmax', name=name + '-predictions')(aux_classifier)
    return aux_classifier


def InceptionV1(input_shape, classes, weight_decay=.0, use_bn=True):
    input = Input(shape=input_shape)
    x = input
    x = conv2d_bn_relu(x,
                       filters=64, kernel_size=(7, 7), name='1a',
                       weight_decay=weight_decay, strides=(2, 2), use_bn=use_bn)
    x = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='1-pool')(x)
    x = conv2d_bn_relu(x,
                       filters=64, kernel_size=(1, 1), weight_decay=weight_decay,
                       name='2a', use_bn=use_bn)
    x = conv2d_bn_relu(x, filters=192, kernel_size=(3, 3), weight_decay=weight_decay,
                       name='2b', use_bn=use_bn)
    x = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='2-pool')(x)

    # inception3a
    x = inception_block_v1(x, (64, (96, 128), (16, 32), 32),
                           weight_decay=weight_decay,
                           name='inception3a', use_bn=use_bn)
    # inception3b
    x = inception_block_v1(x, (128, (128, 192), (32, 96), 64),
                           weight_decay=weight_decay,
                           name='inception3b', use_bn=use_bn)
    x = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='3pool')(x)

    # inception4a
    x = inception_block_v1(x, (192, (96, 208), (16, 48), 64),
                           weight_decay=weight_decay,
                           name='inception4a', use_bn=use_bn)
    # inception4b
    x = inception_block_v1(x, (160, (112, 224), (24, 64), 64),
                           weight_decay=weight_decay,
                           name='inception4b', use_bn=True)
    # inception4c
    x = inception_block_v1(x, (128, (128, 256), (24, 64), 64),
                           weight_decay=weight_decay,
                           name='inception4c', use_bn=use_bn)
    # inception4d
    x = inception_block_v1(x, (112, (144, 288), (32, 64), 64),
                           weight_decay=weight_decay,
                           name='inception4d', use_bn=use_bn)
    # inception4e
    x = inception_block_v1(x, (256, (160, 320), (32, 128), 128),
                           weight_decay=weight_decay,
                           name='inception4e', use_bn=use_bn)
    x = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='4pool')(x)

    # inception5a
    x = inception_block_v1(x, (256, (160, 320), (32, 128), 128),
                           weight_decay=weight_decay,
                           name='inception5a', use_bn=use_bn)
    # inception5b
    x = inception_block_v1(x, (384, (192, 384), (48, 128), 128),
                           weight_decay=weight_decay,
                           name='inception5b', use_bn=use_bn)
    # average pool
    x = AveragePooling2D(pool_size=(7, 7), strides=(1, 1), padding='valid', name='avg7x7')(x)
    x = Dropout(0.4)(x)
    x = Flatten(name='flatten')(x)
    x = Dense(classes, activation='softmax', name='predictions')(x)
    model = Model(input, x, name='inception_v1')
    return model


def InceptionV1ForCIFAR10(input_shape, classes, weight_decay=.0, use_bn=True):
    input = Input(shape=input_shape)
    x = input
    # x = conv2d_bn_relu(x,
    #                    filters=64, kernel_size=(7, 7), name='1a',
    #                    weight_decay=weight_decay, strides=(2, 2), use_bn=use_bn)
    # x = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='1-pool')(x)
    x = conv2d_bn_relu(x,
                       filters=64, kernel_size=(1, 1), weight_decay=weight_decay,
                       name='2a', use_bn=use_bn)
    x = conv2d_bn_relu(x, filters=192, kernel_size=(3, 3), weight_decay=weight_decay,
                       name='2b', use_bn=use_bn)
    # x = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='2-pool')(x)

    # inception3a
    x = inception_block_v1(x, (64, (96, 128), (16, 32), 32),
                           weight_decay=weight_decay,
                           name='inception3a', use_bn=use_bn)
    # inception3b
    x = inception_block_v1(x, (128, (128, 192), (32, 96), 64),
                           weight_decay=weight_decay,
                           name='inception3b', use_bn=use_bn)
    x = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='3pool')(x)

    # inception4a
    x = inception_block_v1(x, (192, (96, 208), (16, 48), 64),
                           weight_decay=weight_decay,
                           name='inception4a', use_bn=use_bn)
    # inception4b
    x = inception_block_v1(x, (160, (112, 224), (24, 64), 64),
                           weight_decay=weight_decay,
                           name='inception4b', use_bn=True)
    # inception4c
    x = inception_block_v1(x, (128, (128, 256), (24, 64), 64),
                           weight_decay=weight_decay,
                           name='inception4c', use_bn=use_bn)
    # inception4d
    x = inception_block_v1(x, (112, (144, 288), (32, 64), 64),
                           weight_decay=weight_decay,
                           name='inception4d', use_bn=use_bn)
    # inception4e
    x = inception_block_v1(x, (256, (160, 320), (32, 128), 128),
                           weight_decay=weight_decay,
                           name='inception4e', use_bn=use_bn)
    x = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='4pool')(x)

    # inception5a
    x = inception_block_v1(x, (256, (160, 320), (32, 128), 128),
                           weight_decay=weight_decay,
                           name='inception5a', use_bn=use_bn)
    # inception5b
    x = inception_block_v1(x, (384, (192, 384), (48, 128), 128),
                           weight_decay=weight_decay,
                           name='inception5b', use_bn=use_bn)

    # average pool
    x = AveragePooling2D(pool_size=(8, 8), strides=(1, 1), padding='valid', name='avg8x8')(x)
    # x = Dropout(0.4)(x)
    x = Flatten(name='flatten')(x)
    x = Dense(classes, activation='softmax', name='predictions')(x)
    model = Model(input, x, name='inception_v1')
    return model
