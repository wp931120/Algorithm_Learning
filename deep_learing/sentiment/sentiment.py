from keras.layers import Dense, Flatten
from keras.layers import Embedding, Dropout, LSTM, GRU, Bidirectional
from keras.models import Model
from attention import AttentionLayer
from keras.layers import Input,GlobalAvgPool1D
from keras.preprocessing import sequence
from keras.datasets import imdb
import keras.backend as K

max_features = 20000
Maxlen = 80
batch_size = 32

def load_data(maxlen = Maxlen):
    """
    load imdb data ,and padding the sequence to max length
    :param maxlen:
    :return:
    """
    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
    train_data = sequence .pad_sequences(train_data, maxlen=maxlen)
    test_data = sequence .pad_sequences(test_data, maxlen=maxlen)
    return train_data, train_labels, test_data, test_labels


def model(max_features,maxlen,attention=True):
    """
    build a model with bi-gru ,you also can choose add attention layer or not
    you should define the max_features and maxlen
    :param max_features:
    :param maxlen:
    :param attention:
    :return:
    """
    embedding_layer = Embedding(input_dim=max_features,
                                output_dim=128,
                                input_length=maxlen,
                                trainable=True)

    sequence_input = Input(shape=(maxlen,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    gru = Bidirectional(GRU(100, return_sequences=True))(embedded_sequences)
    if attention:
        att = AttentionLayer()(gru)
        preds = Dense(1, activation='sigmoid')(att)
    else:
        flat = GlobalAvgPool1D()(gru)
        preds = Dense(1, activation='sigmoid')(flat)
    model = Model(sequence_input, preds)
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc'])

    model.summary()
    return model


def train(model, train_data, train_labels, test_data, test_labels, epochs=5, batch_size=32):
    """
    train the model and define the hyperparameters
    :param model:
    :param train_data:
    :param train_labels:
    :param test_data:
    :param test_labels:
    :param epochs:
    :param batch_size:
    :return:
    """
    model.fit(train_data, train_labels,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(test_data, test_labels))


if __name__ =="__main__":
    train_data, train_labels, test_data, test_labels = load_data(maxlen=Maxlen)

    # without attention layer
    model = model(max_features=max_features, maxlen=Maxlen, attention=False)
    train(model, train_data, train_labels, test_data, test_labels)

    K.clear_session()
    # with attention layer
    model = model(max_features=max_features, maxlen=Maxlen, attention=True)
    train(model, train_data, train_labels, test_data, test_labels)