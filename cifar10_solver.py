from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import keras.backend as K


def plot_history(history):
    """
    plot train epoch history and acc
    :param history: train history object returned by CIFAR10Solver.train()
    """
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.xlabel('epoch')
    plt.ylabel('Loss value')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.xlabel('epoch')
    plt.ylabel('acc value')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


class CIFAR10Solver(object):
    """
    A CIFAR10Solver encapsulates all the logic nessary for training cifar10 classifiers.The train model is defined
    outside, you must pass it to init().

    The solver train the model, plot loss and aac history, and test on the test data.

    Example usage might look something like this.

    model = MyAwesomeModel(opt=SGD, losses='categorical_crossentropy',  metrics=['acc'])
    model.compile(...)
    model.summary()
    solver = CIFAR10Solver(model)
    history = solver.train()
    plotHistory(history)
    solver.test()
    """

    def __init__(self, model, data):
        """

        :param model: A model object conforming to the API described above
        :param data:  A tuple of training, validation and test data from CIFAR10Data
        """
        self.model = model
        self.X_train, self.Y_train, self.X_val, self.Y_val, self.X_test, self.Y_test = data

    def __on_epoch_end(self, epoch, logs=None):
        print(K.eval(self.model.optimizer.lr))

    def train(self, epochs=200, batch_size=128, data_augmentation=True, callbacks=None):
        if data_augmentation:
            # datagen
            datagen = ImageDataGenerator(
                featurewise_center=False,  # set input mean to 0 over the dataset
                samplewise_center=False,  # set each sample mean to 0
                featurewise_std_normalization=False,  # divide inputs by std of the dataset
                samplewise_std_normalization=False,  # divide each input by its std
                zca_whitening=False,  # apply ZCA whitening
                # rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
                width_shift_range=4,  # randomly shift images horizontally (fraction of total width)
                height_shift_range=4,  # randomly shift images vertically (fraction of total height)
                horizontal_flip=True,  # randomly flip images
                vertical_flip=False,  # randomly flip images
            )
            # (std, mean, and principal components if ZCA whitening is applied).
            # datagen.fit(x_train)
            print('train with data augmentation')
            train_gen = datagen.flow(self.X_train, self.Y_train, batch_size=batch_size)
            history = self.model.fit_generator(generator=train_gen,
                                               epochs=epochs,
                                               callbacks=callbacks,
                                               validation_data=(self.X_val, self.Y_val),
                                               )
        else:
            print('train without data augmentation')
            history = self.model.fit(self.X_train, self.Y_train,
                                     batch_size=batch_size, epochs=epochs,
                                     callbacks=callbacks,
                                     validation_data=(self.X_val, self.Y_val),
                                     )
        return history

    def test(self):
        loss, acc = self.model.evaluate(self.X_test, self.Y_test)
        print('test data loss:%.2f acc:%.4f' % (loss, acc))
