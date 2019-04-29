from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import Conv2D

import tensorflow as tf

class Augmented_Conv2D(Layer):

    def __init__(self, filters, kernel_size, dk, dv, Nh,relative,**kwargs):
        self.filters = filters
        self.kernel_size = kernel_size
        self.dk = dk
        self.dv = dv
        self.Nh = Nh
        self.relative = relative
        self.conv_out = Conv2D(self.filters-self.dv,
                               kernel_size=self.kernel_size)
        self.qkv_conv = Conv2D(2 * self.dk + self.dv, kernel_size=1)
        self.attn_out = Conv2D(self.dv, kernel_size=1)
        super(Augmented_Conv2D, self).__init__(**kwargs)

    def build(self, input_shape):
        self.batch_size, self.channels, self.height, self.width = input_shape
        super(Augmented_Conv2D, self).build(input_shape)

    def call(self, x):
        conv_out = self.conv_out(x)
        flat_q, flat_k, flat_v, q, k, v = self.compute_flat_qkv(x, self.dk, self.dv, self.Nh)

        flat_q = tf.transpose(flat_q,[0,1,3,2])

        logits = tf.matmul(flat_q, flat_k)

        if self.relative:
            h_rel_logits, w_rel_logits = self.relative_logits(q)
            logits += h_rel_logits
            logits += w_rel_logits
        weights = K.softmax(logits, axis=-1)

        # attn_out
        # (batch, Nh, height * width, dv)
        flat_v = tf.transpose(flat_v, [0, 1, 3, 2])
        attn_out = tf.matmul(weights, flat_v)
        attn_out = tf.reshape(attn_out, (-1, self.Nh, self.dv // self.Nh, self.height, self.width))
        # combine_heads_2d
        # (batch, out_channels, height, width)
        attn_out = self.combine_heads_2d(attn_out)
        attn_out = self.attn_out(attn_out)
        print(attn_out.shape,conv_out.shape)
        return tf.concat((conv_out, attn_out), axis=1)

    def split_heads_2d(self,inputs, Nh):
        batch, channels, height, width = inputs.get_shape().as_list()
        ret_shape = [-1, Nh, channels // Nh, height, width]
        split = tf.reshape(inputs, ret_shape)
        return tf.transpose(split, [0, 3, 1, 2, 4])

    def compute_flat_qkv(self,inputs, dk, dv, Nh):
        N, _, H, W = inputs.get_shape().as_list()
        qkv = self.qkv_conv(inputs)
        q, k, v = tf.split(qkv, [dk, dk, dv], axis=3)
        q = self.split_heads_2d(q, Nh)
        k = self.split_heads_2d(k, Nh)
        v = self.split_heads_2d(v, Nh)
        dkh = dk // Nh
        q *= dkh ** -0.5
        flat_q = tf.reshape(q, [-1, Nh, H * W, dk])
        flat_k = tf.reshape(k, [-1, Nh, H * W, dk])
        flat_v = tf.reshape(v, [-1, Nh, H * W, dv])
        return flat_q, flat_k, flat_v, q, k, v

    def combine_heads_2d(self, x):
        batch, Nh, dv, H, W = x.get_shape().as_list()
        ret_shape = (-1, Nh * dv, H, W)
        return tf.reshape(x, ret_shape)

    def relative_logits(self,q):
        B, Nh, dk, H, W = q.get_shape().as_list()
        q = tf.transpose(q,[0, 1, 4, 3, 2])
        q = tf.transpose(q, [0, 1, 3, 2, 4])
        key_rel_w = tf.get_variable(
            'key_rel_w', shape=(2 * W-1, dk),
            initializer=tf.random_normal_initializer(dk **-0.5),
            trainable=True
        )
        rel_logits_w = self.relative_logits_1d(q,key_rel_w, H, W,Nh,"w")
        key_rel_h = tf.get_variable(
            'key_rel_h', shape=(2 * H-1, dk),
            initializer=tf.random_normal_initializer(dk **-0.5),
            trainable=True
        )
        rel_logits_h = self.relative_logits_1d(tf.transpose(q, [0, 1, 3, 2, 4]),
                                               key_rel_h, W, H, Nh,"h")
        return rel_logits_h, rel_logits_w

    def relative_logits_1d(self, q, rel_k, H, W, Nh, case):
        rel_logits = tf.einsum('bhxyd,md->bhmxy', q, rel_k)
        rel_logits = tf.reshape(rel_logits, (-1, Nh * H, W, 2 * W - 1))
        rel_logits = self.rel_to_abs(rel_logits)

        rel_logits = tf.reshape(rel_logits, (-1, Nh, H, W, W))
        rel_logits = tf.expand_dims(rel_logits, axis=3)
        rel_logits = tf.tile(rel_logits,(1, 1, 1, H, 1, 1))


        if case is "w":
            rel_logits = tf.transpose(rel_logits, [0 ,1, 2, 4, 3, 5])
        elif case is "h":
            rel_logits = tf.transpose(rel_logits,[0, 1, 4, 3, 2, 5])
            rel_logits = tf.transpose(rel_logits,[0, 1, 2, 3, 5, 4])
            rel_logits = tf.transpose(rel_logits, [0, 1, 2, 5, 4, 3])
        rel_logits = tf.reshape(rel_logits, (-1, Nh, H * W, H * W))
        return rel_logits

    def rel_to_abs(self, x):
        B, Nh, L, _ = x.get_shape().as_list()

        col_pad = tf.zeros_like(x[:,:,:,:1])
        x = tf.concat((x, col_pad), axis=3)

        flat_x = tf.reshape(x, (-1, Nh, L * 2 * L))
        flat_pad = tf.zeros_like(x[:,:,:L-1,1])
        flat_x_padded = tf.concat((flat_x, flat_pad), axis=2)

        final_x = tf.reshape(flat_x_padded, (-1, Nh, L + 1, 2 * L - 1))
        final_x = final_x[:, :, :L, L - 1:]
        return final_x

    def compute_output_shape(self, input_shape):
        batch_size, channels, height, width = input_shape
        output_dim = self.filters
        return (batch_size,output_dim,height,width)


if __name__ == '__main__':
    from keras.layers import Input
    from keras.models import Model
    input = Input(shape=(64,28,28))
    attention_augmented_conv = Augmented_Conv2D(filters=64,kernel_size=3,
                                                dk=32,dv=32,Nh=4,relative=False)(input)
    model = Model(inputs=input, outputs=attention_augmented_conv)
    print(model.summary())