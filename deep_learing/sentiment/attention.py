import keras.backend as K
from keras.layers.core import Layer

class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        # W.shape = (time_steps, time_steps)
        self.W = self.add_weight(name='att_weight',
                                 shape=(input_shape[1], input_shape[1]),
                                 initializer='uniform',
                                 trainable=True)
        self.b = self.add_weight(name='att_bias',
                                 shape=(input_shape[1],),
                                 initializer='uniform',
                                 trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        # inputs.shape = (batch_size, time_steps, seq_len)
        x = K.permute_dimensions(inputs, (0, 2, 1))
        ##################################################################
        # x.shape = (batch_size, seq_len, time_steps)
        # W.shape = (time_steps,time_steps) b.shape(time_steps)
        a = K.softmax(K.tanh(K.dot(x, self.W) + self.b))
        # a.shape = x.shape = (batch_size, seq_len, time_steps)
        outputs = K.permute_dimensions(a * x, (0, 2, 1))
        # outputs.shape = inputs.shape = (batch_size, seq_len, time_steps)
        outputs = K.sum(outputs, axis=1)
        # outputs.shape=(batch_size,time_steps)
        ###################################################################
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[2]
