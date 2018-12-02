import numpy as np
import keras
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import Conv2D, Dense
from keras.layers import MaxPooling2D, Dropout, Flatten
from keras.datasets import cifar100
from keras.utils import np_utils
"""
BN层：（1）防止梯度消失（2）加速模型训练 
Batch normalization减轻了输入的变化的影响。通过对神经元的输出进行归一化，激活函数的输入都是在0附近的，这就保证了没有梯度的消失。

Batch normalization将每一层的输出变换成一个单位的高斯分布。由于这些输出被输入到一个激活函数中，激活后的值也是一个正态的分布。
因为一层的输出是下一层的输入，每一层的输入的分布对于不同的batch来说就不会有太大的变化。通过减小输入层的分布的变化，使得训练速度加快，得到更加准确的结果。
"""

#load the data
(x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')


class CNN:

    def __init__(self, x_train, y_train, x_test, y_test):
        """
        do some simple data processing
        :param x_train:
        :param y_train:
        :param x_test:
        :param y_test:
        """
        self.x_train = x_train - np.mean(x_train)
        self.y_train = np_utils.to_categorical(y_train)
        self.x_test = x_test - x_test.mean()
        self.y_test = np_utils.to_categorical(y_test)

    def data_generate(self):
        """
        use keras API to generate data flow and do real time data argumentation
        :return:
        """
        train_datagen = ImageDataGenerator(
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)
        train_datagen.fit(self.x_train)
        train_generator = train_datagen.flow(self.x_train,
                                             y=self.y_train,
                                             batch_size=80, )
        return train_generator

    def conv_block(self, model, bn=True, activation="relu"):
        """
        define convolution layer which you can choose if add batch normaliztion layer
        :param model:
        :param bn:
        :param activation:
        :return:
        """
        model.add(Conv2D(60, 3, padding="same", input_shape=x_train.shape[1:]))
        if bn:
            model.add(BatchNormalization())
        model.add(Activation(activation))
        # Second Stacked Convolution
        model.add(Conv2D(30, 3, padding="same"))
        if bn:
            model.add(BatchNormalization())
        model.add(Activation(activation))

        model.add(MaxPooling2D())
        model.add(Dropout(0.15))
        return model

    def fn_layer(self, model, category_num=100):
        """
        define the fully connected layer
        :param model:
        :return:
        """
        model.add(Dense(category_num, activation="softmax"))
        return model


    def build_model(self, optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"], bn=True, activation="relu"):
        """
        to build the model and you can choose the optimizer, loss function,and metrics method
        :param optimizer:
        :param loss:
        :param metrics:
        :param bn:
        :param activation:
        :return:
        """
        model = Sequential()
        model = self.conv_block(model, bn=bn, activation=activation)
        model.add(Flatten())
        model = self.fn_layer(model)
        model.compile(optimizer=optimizer,
                      loss=loss,
                      metrics=metrics)
        model.summary()
        return model

    def trian_model(self, model, epochs=10, callbacks=None):
        """
        train the model
        :param model:
        :param epochs:
        :param callbacks:
        :return:
        """
        model.fit_generator(
            self.data_generate(),
            steps_per_epoch=625,
            epochs=epochs,
            verbose=1,
            validation_data=(self.x_test, self.y_test),
            callbacks=callbacks
        )


class LossHistory(keras.callbacks.Callback):
    """
    to define a class to draw the loss and accuracy of train and val you can put the class
    into callback function when train the model
    """
    def on_train_begin(self, logs={}):
        self.losses = {'batch': [], 'epoch': []}
        self.accuracy = {'batch': [], 'epoch': []}
        self.val_loss = {'batch': [], 'epoch': []}
        self.val_acc = {'batch': [], 'epoch': []}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.show()


if __name__ == "__main__":
    """
    do the experiment 
    """
    cnn = CNN(x_train, y_train, x_test, y_test)
    history = LossHistory()
    model_bn = cnn.build_model()
    cnn.trian_model(model_bn, callbacks=[history])
    history.loss_plot('epoch')
    model_nobn = cnn.build_model(bn=False)
    cnn.trian_model(model_nobn, callbacks=[history])
    history.loss_plot('epoch')



