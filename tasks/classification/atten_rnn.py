from keras import Input, Model
from keras.layers import Embedding, LSTM, Dense, Bidirectional,GlobalAveragePooling1D
from base.base_model import BaseModel
import numpy as np
from nlpy.layers.attention import SeqSelfAttention
from nlpy.layers import MaskedGlobalAveragePooling1D
import keras


class Attention_RNN(BaseModel):
    def __init__(self, config, pretrain_embedding=None):
        super(Attention_RNN, self).__init__(config)
        self.model = self.build_model(pretrain_embedding)

    def build_model(self, pretrain_embedding=None):
        input = Input((self.config.model.sequence_length,))

        embedding = Embedding(self.config.model.vocabulary_size + 1,
                              self.config.model.embedding_dim,
                              weights=[pretrain_embedding] if type(pretrain_embedding) == np.ndarray else None,
                              input_length=self.config.model.sequence_length,
                              trainable=self.config.model.embedding_trainable
                              )(input)
        if self.config.model.bidirectional:
            x = Bidirectional(LSTM(self.config.model.rnn_hidden_size, return_sequences=True))(embedding)
        else:
            x = LSTM(self.config.model.rnn_hidden_size, return_sequences=True)(embedding)
        attention = SeqSelfAttention(attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,
                                     kernel_regularizer=keras.regularizers.l2(1e-4),
                                     bias_regularizer=keras.regularizers.l1(1e-4),
                                     attention_regularizer_weight=1e-4,
                                     name='Attention')(x)
        attention = GlobalAveragePooling1D()(attention)
        output = Dense(self.config.model.class_num,
                       activation=self.config.model.last_activation)(attention)
        model = Model(inputs=input, outputs=output)
        print(model.summary(line_length=100))
        return model
