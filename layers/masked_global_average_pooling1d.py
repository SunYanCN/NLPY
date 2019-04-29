import keras.backend as K
from keras.layers.pooling import _GlobalPooling1D

class MaskedGlobalAveragePooling1D(_GlobalPooling1D):

    def __init__(self, **kwargs):
        self.support_mask = True
        super(MaskedGlobalAveragePooling1D, self).__init__(**kwargs)

    def build(self, input_shape):
        super(MaskedGlobalAveragePooling1D, self).build(input_shape)
        self.feat_dim = input_shape[2]

    def call(self, x, mask=None):
        if mask is None:
            return K.mean(x, axis=1)
        mask = K.cast(mask, "float32")
        expanded_mask = K.expand_dims(mask)
        x_masked = x * expanded_mask
        mask_counts = K.sum(mask, axis=-1)
        x_sums = K.sum(x_masked, axis=1)
        counts_cast = K.expand_dims(mask_counts)
        return x_sums / counts_cast

    def compute_mask(self, input_shape, input_mask=None):
        return None