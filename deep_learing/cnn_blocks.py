from keras.layers import Conv2D, Dense, MaxPooling2D, \
    AveragePooling2D, GlobalAveragePooling2D
from keras.models import Sequential, Model
from keras.layers import Input, concatenate, add
from keras.layers.core import Flatten
import keras.backend as K


def conv2d(x, f, k=3, s=1, p='same', d=1, a='relu'):
    """
    define a convolution layer
    :param x:
    :param f:
    :param k:
    :param s:
    :param p:
    :param d:
    :param a:
    :return:
    """
    return Conv2D(filters=f, kernel_size=k, strides=s,
                  padding=p, dilation_rate=d, activation=a)(x)


def maxpool(x, k=2, s=2, p='same'):
    """
    define max pooling layer
    :param x:
    :param k:
    :param s:
    :param p:
    :return:
    """
    return MaxPooling2D(pool_size=k, strides=s, padding=p)(x)


def bottleneck(x, f=32, r=4):
    """
    bottleneck block
    :param x:  the input tensor
    :param f:  the filter numbers
    :param r:  the parameter can used to change your filter numbers
    :return:
    """
    x = conv2d(x, f // r, k=1)
    x = conv2d(x, f // (2 * r), k=3)
    x = conv2d(x, f // r, k=3)
    return conv2d(x, f, k=1)


def inception(x, f=32):
    """
    inception block
    :param x: the input tensor
    :param f: the filter numbers
    :return:
    """
    a = conv2d(x, f, k=1)
    b = conv2d(x, f, k=3)
    c = conv2d(x, f, k=5)
    d = maxpool(x, k=3, s=1)
    return concatenate([a, b, c, d])


def residual_block(x, f=32, r=4):
    """
    residual block
    :param x: the input tensor
    :param f: the filter numbers
    :param r:
    :return:
    """
    m = conv2d(x, f // r, k=1)
    m = conv2d(m, f // r, k=3)
    m = conv2d(m, f, k=1)
    return add([x, m])


def dense_block(x, f=32, d=5):
    """
    the dense block
    :param x: the input tensor
    :param f: the filter numbers
    :param d:
    :return:
    """
    l = x
    for i in range(d):
        x = conv2d(l, f)
        l = concatenate([l, x])
    return l


def print_dense():
    inputs = Input(shape=(28, 28, 1))
    x = dense_block(inputs)
    x = GlobalAveragePooling2D()(x)
    x = Dense(64, activation='relu')(x)
    # x = Flatten()(x)
    predictions = Dense(10, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()


def print_residual():
    K.clear_session()
    inputs = Input(shape=(28, 28, 1))
    x = residual_block(inputs)
    x = GlobalAveragePooling2D()(x)
    x = Dense(64, activation='relu')(x)
    # x = Flatten()(x)
    predictions = Dense(10, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()


def print_bottleneck():
    K.clear_session()
    inputs = Input(shape=(28, 28, 1))
    x = bottleneck(inputs)
    x = GlobalAveragePooling2D()(x)
    x = Dense(64, activation='relu')(x)
    # x = Flatten()(x)
    predictions = Dense(10, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()


def print_inception():
    K.clear_session()
    inputs = Input(shape=(28, 28, 1))
    x = inception(inputs)
    x = GlobalAveragePooling2D()(x)
    x = Dense(64, activation='relu')(x)
    predictions = Dense(10, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()


if __name__ == "__main__":
    pass






