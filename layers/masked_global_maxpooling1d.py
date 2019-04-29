import keras.backend as K
from keras.layers.pooling import _GlobalPooling1D

class MaskedGlobalMaxPooling1D(_GlobalPooling1D):

    def __init__(self, **kwargs):
        self.support_mask = True
        super(MaskedGlobalMaxPooling1D, self).__init__(**kwargs)

    def build(self, input_shape):
        super(MaskedGlobalMaxPooling1D, self).build(input_shape)
        self.feat_dim = input_shape[2]

    def call(self, x, mask=None):
        if mask is None:
            return K.max(x, axis=1)
        mask = K.cast(mask, "float32")
        expanded_mask = K.expand_dims(mask)
        x_masked = x * expanded_mask
        return K.max(x_masked, axis=1)

    def compute_mask(self, input_shape, input_mask=None):
        return None