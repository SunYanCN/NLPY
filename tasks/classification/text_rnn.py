from keras import Input, Model
from keras.layers import Embedding,CuDNNLSTM, Dense,Bidirectional,SpatialDropout1D,LSTM
from base.base_model import BaseModel
import numpy as np

class TextRNN(BaseModel):
    def __init__(self,config,pretrain_embedding=None):
        super(TextRNN,self).__init__(config)
        self.model = self.build_model(pretrain_embedding)

    def build_model(self, pretrain_embedding=None):
        input = Input((self.config.model.sequence_length,))

        embedding = Embedding(self.config.model.vocabulary_size+1,
                              self.config.model.embedding_dim,
                              weights=[pretrain_embedding] if type(pretrain_embedding) == np.ndarray else None,
                              input_length=self.config.model.sequence_length,
                              trainable=self.config.model.embedding_trainable,
                              mask_zero=True
                              )(input)
        # embedding = SpatialDropout1D(0.2, seed=1234)(embedding)
        if self.config.model.bidirectional:
            x = Bidirectional(LSTM(self.config.model.rnn_hidden_size))(embedding)
        else:
            x = LSTM(self.config.model.rnn_hidden_size)(embedding)
        output = Dense(self.config.model.class_num,
                       activation=self.config.model.last_activation)(x)
        model = Model(inputs=input, outputs=output)
        print(model.summary(line_length=100))
        return model