import keras
from keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np


class CIFAR10Data(object):
    def __init__(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = cifar10.load_data()
        self.classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        print('Training data shape:', self.x_train.shape)
        print('Training label shape', self.y_train.shape)
        print('Test data shape', self.x_test.shape)
        print('Test label shape', self.y_test.shape)

    def get_stretch_data(self, subtract_mean=True):
        """
        reshape X each image to row vector, and transform Y to one_hot label.
        :param subtract_mean:Indicate whether subtract mean image.
        :return: x_train, one_hot_y_train, x_test, one_hot_y_test
        """
        num_classes = len(self.classes)
        x_train = np.reshape(self.x_train, (self.x_train.shape[0], -1)).astype('float64')
        y_train = keras.utils.to_categorical(self.y_train, num_classes)

        x_test = np.reshape(self.x_test, (self.x_test.shape[0], -1)).astype('float64')
        y_test = keras.utils.to_categorical(self.y_test, num_classes)

        if subtract_mean:
            mean_image = np.mean(x_train, axis=0)
            x_train -= mean_image
            x_test -= mean_image
            # print(x_mean[:10])
            # plt.figure(figsize=(4, 4))
            # plt.imshow(x_mean.reshape((32, 32, 3)))
            # plt.show()

        return x_train, y_train, x_test, y_test

    def get_data(self, subtract_mean=True):
        """
        The data is not reshaped, keep 3 channel.
        :param subtract_mean:Indicate whether subtract mean image.
        :return: x_train, one_hot_y_train, x_test, one_hot_y_test
        """
        num_classes = len(self.classes)
        x_train = self.x_train.astype('float64')
        y_train = keras.utils.to_categorical(self.y_train, num_classes)

        x_test = self.x_test.astype('float64')
        y_test = keras.utils.to_categorical(self.y_test, num_classes)

        if subtract_mean:
            mean_image = np.mean(x_train, axis=0)
            x_train -= mean_image
            x_test -= mean_image
        return x_train, y_train, x_test, y_test


def plot_cifar10(cifar_data, num_sample_per_class):
    """
    random select num_sample_per_class to plot
    """
    num_classes = len(cifar_data.classes)

    plt.figure()
    for y, cls in enumerate(cifar_data.classes):
        cls_indices = np.flatnonzero(cifar_data.y_train == y)
        samples_indices = np.random.choice(cls_indices, num_sample_per_class, replace=False)
        samples = cifar_data.x_train[samples_indices]
        for x, sample in enumerate(samples):
            # subplot index count from 1
            plt_idx = x * num_classes + y + 1
            plt.subplot(num_sample_per_class, num_classes, plt_idx)
            plt.imshow(sample)
            plt.axis('off')
            if x == 0:
                plt.title(cls)
    plt.show()