import tensorflow as tf
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Conv2DTranspose, LeakyReLU, Activation, Dropout, BatchNormalization, LeakyReLU, GlobalMaxPool2D
import config


# # TODO: [conv overlapping model]//layers and width parameter tune
# def define_descrminator(image_size):
#     init = RandomNormal(stddev=0.02)
#     input_img = Input(shape=image_size)
#     # C64
#     d = Conv2D(64, (4, 4), (2, 2), padding='SAME', kernel_initializer=init)(input_img)
#     d = LeakyReLU(alpha=0.2)(d)
# 	# C128
#     d = Conv2D(128, (4, 4), (2, 2), padding='SAME', kernel_initializer=init)(d)
#     d = BatchNormalization()(d)
#     d = LeakyReLU(alpha=0.2)(d)
# 	# C256
#     d = Conv2D(256, (4, 4), (2, 2), padding='SAME', kernel_initializer=init)(d)
#     d = BatchNormalization()(d)
#     d = LeakyReLU(alpha=0.2)(d)
#     # flatten
#     flt = Flatten()(d)
#     # linear logits layer
#     output = Dense(1)(flt)
#     #build and compile the model
#     model = Model(inputs=input_img, outputs=output, name='style_descriminator')
#     return model

# # TODO: [conv non-overlapping model]//layers and width parameter tune
# def define_descrminator(image_size):
#     init = RandomNormal(stddev=0.02)
#     input_img = Input(shape=image_size)
#     # C64
#     d = Conv2D(64, (4, 4), (4, 4), padding='SAME', kernel_initializer=init)(input_img)
#     d = LeakyReLU(alpha=0.2)(d)
# 	# C128
#     d = Conv2D(128, (4, 4), (4, 4), padding='SAME', kernel_initializer=init)(d)
#     d = BatchNormalization()(d)
#     d = LeakyReLU(alpha=0.2)(d)
# 	# C256
#     d = Conv2D(256, (4, 4), (4, 4), padding='SAME', kernel_initializer=init)(d)
#     d = BatchNormalization()(d)
#     d = LeakyReLU(alpha=0.2)(d)
#     # flatten
#     flt = Flatten()(d)
#     # linear logits layer
#     output = Dense(1)(flt)
#     #build and compile the model
#     model = Model(inputs=input_img, outputs=output, name='style_descriminator')
#     return model

def define_style_descrminator(output_size,image_shape):
    vgg16_feature_net = tf.keras.applications.VGG16(include_top=False, input_shape=image_shape)
    for layer in vgg16_feature_net.layers:
        layer.trainable = False
    feature_map = vgg16_feature_net.output
    x = GlobalMaxPool2D()(feature_map)
    x = Dropout(rate=0.5)(x)
    x = Dense(512, activation='relu')(x)
    logits = Dense(output_size, use_bias=False)(x)
    model = Model(inputs=vgg16_feature_net.input, outputs=logits)
    return model

class StyleNet(tf.keras.Model):

    def __init__(self, base_model):
        super(StyleNet, self).__init__()
        self._model = base_model 

    @tf.function
    def call(self, inputs):
        ref_img, trans_img = inputs
        with tf.name_scope("Style") as scope:
            ft1 = self._model(ref_img)
            ft1 = tf.math.l2_normalize(ft1, axis=-1)
        with tf.name_scope("Transfer") as scope:
            ft2 = self._model(trans_img)
            ft2 = tf.math.l2_normalize(ft2, axis=-1)
        return [ft1, ft2]
    
    @tf.function
    def get_features(self, inputs):
        return tf.math.l2_normalize(self._model(inputs), axis=-1)

#base_model = define_style_descrminator(config.DESCS_LATENT_SIZE, config.IMAGE_SHAPE)

class RankingMetrics(tf.keras.metrics.Metric):

    def __init__(self, name='pairwise_ranking_loss', **kwargs):
        super(RankingMetrics, self).__init__(name=name, **kwargs)
        
        self.relative_acc = self.add_weight(name='rel_acc', initializer='zeros')
        self.ranking_loss = self.add_weight(name='ranking_loss', initializer='zeros')

    def update_state(self, y_ref, y_style, label):
        m  = tf.cast(tf.broadcast_to(config.DESC_ALPHA, shape=y_ref.shape), dtype=tf.float32)
        u  = tf.cast(tf.broadcast_to(0, shape=y_ref.shape), dtype=tf.float32)
        i  = tf.cast(tf.broadcast_to(1, shape=y_ref.shape), dtype=tf.float32)
        y = tf.cast(label[..., tf.newaxis], dtype=tf.float32)
        dist = tf.norm(y_ref-y_style, ord='euclidean', axis=-1, keepdims=True)
        loss = y*dist + (i-y)*tf.reduce_max(tf.stack([u,m-dist]), axis=0)
        rk_loss = tf.reduce_mean(loss)
        self.ranking_loss.assign(rk_loss)

        prb = dist/m
        bool_prb = tf.cast(tf.math.floor(prb), dtype=tf.bool)
        bool_lbl = tf.cast(label, dtype=tf.bool)
        rel_acc = tf.reduce_mean(tf.cast(tf.math.logical_not(tf.math.logical_xor(bool_prb, bool_lbl)), dtype=tf.float32))
        self.relative_acc.assign(rel_acc)
        


    def result(self):
        return self.relative_acc, self.ranking_loss
