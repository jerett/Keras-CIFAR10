from keras.models import Model
from keras.models import Input
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Conv2D
from keras.layers import MaxPool2D
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras import regularizers


def conv2d_bn_relu(model,
                   filters,
                   block_index, layer_index,
                   weight_decay=.0, padding='same'):
    conv_name = 'conv' + str(block_index) + '-' + str(layer_index)
    model = Conv2D(filters=filters,
                   kernel_size=(3, 3),
                   padding=padding,
                   kernel_regularizer=regularizers.l2(weight_decay),
                   strides=(1, 1),
                   name=conv_name,
                   )(model)
    bn_name = 'bn' + str(block_index) + '-' + str(layer_index)
    model = BatchNormalization(name=bn_name)(model)
    relu_name = 'relu' + str(block_index) + '-' + str(layer_index)
    model = Activation('relu', name=relu_name)(model)
    return model


def dense2d_bn_dropout(model, units, weight_decay, name):
    model = Dense(units,
                  kernel_regularizer=regularizers.l2(weight_decay),
                  name=name,
                  )(model)
    model = BatchNormalization(name=name + '-bn')(model)
    model = Activation('relu', name=name + '-relu')(model)
    model = Dropout(0.5, name=name + '-dropout')(model)
    return model


def VGGNet(classes, input_shape, weight_decay,
           conv_block_num=5,
           fc_layers=2, fc_units=4096):
    input = Input(shape=input_shape)
    # block 1
    x = conv2d_bn_relu(model=input,
                       filters=64,
                       block_index=1, layer_index=1,
                       weight_decay=weight_decay
                       )
    x = conv2d_bn_relu(model=x,
                       filters=64,
                       block_index=1, layer_index=2,
                       weight_decay=weight_decay)
    x = MaxPool2D(name='pool1')(x)

    # block 2
    if conv_block_num >= 2:
        x = conv2d_bn_relu(x,
                           filters=128,
                           block_index=2, layer_index=1,
                           weight_decay=weight_decay)
        x = conv2d_bn_relu(x,
                           filters=128,
                           block_index=2, layer_index=2,
                           weight_decay=weight_decay)
        x = MaxPool2D(name='pool2')(x)

    # block 3
    if conv_block_num >= 3:
        x = conv2d_bn_relu(x,
                           filters=256,
                           block_index=3, layer_index=1,
                           weight_decay=weight_decay)
        x = conv2d_bn_relu(x,
                           filters=256,
                           block_index=3, layer_index=2,
                           weight_decay=weight_decay)
        x = conv2d_bn_relu(x,
                           filters=256,
                           block_index=3, layer_index=3,
                           weight_decay=weight_decay)
        x = MaxPool2D(name='pool3')(x)

    # block 4
    if conv_block_num >= 4:
        x = conv2d_bn_relu(x,
                           filters=512,
                           block_index=4, layer_index=1,
                           weight_decay=weight_decay)
        x = conv2d_bn_relu(x,
                           filters=512,
                           block_index=4, layer_index=2,
                           weight_decay=weight_decay)
        x = conv2d_bn_relu(x,
                           filters=512,
                           block_index=4, layer_index=3,
                           weight_decay=weight_decay)
        x = MaxPool2D(name='pool4')(x)

    # block 5
    if conv_block_num >= 5:
        x = conv2d_bn_relu(x,
                           filters=512,
                           block_index=5, layer_index=1,
                           weight_decay=weight_decay)
        x = conv2d_bn_relu(x,
                           filters=512,
                           block_index=5, layer_index=2,
                           weight_decay=weight_decay)
        x = conv2d_bn_relu(x,
                           filters=512,
                           block_index=5, layer_index=3,
                           weight_decay=weight_decay)
        x = MaxPool2D(name='pool5')(x)

    x = Flatten(name='flatten')(x)
    if fc_layers >= 1:
        x = dense2d_bn_dropout(x, fc_units, weight_decay, 'fc6')
        if fc_layers >= 2:
            x = dense2d_bn_dropout(x, fc_units, weight_decay, 'fc7')
    out = Dense(classes, activation='softmax', name='predictions')(x)
    model = Model(input, out)
    return model


def VGG13(classes):
    return VGGNet(classes, weight_decay=1e-4, conv_block_num=4, fc_layers=2, fc_units=4096)


def VGG16(classes):
    return VGGNet(classes, weight_decay=1e-4, conv_block_num=5, fc_layers=2, fc_units=4096)