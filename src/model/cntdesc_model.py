#%%
import tensorflow as tf  
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal, HeUniform
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Conv2DTranspose, LeakyReLU, Activation, Dropout, BatchNormalization, LeakyReLU, GlobalMaxPool2D, Concatenate, ReLU, AveragePooling2D
from tensorflow.keras import losses
from tensorflow.keras import metrics
#import config 

#%%
def define_cnt_encoder(latent_size, image_shape=(128, 128, 3)):
    #content image input
    img_in = Input(shape=image_shape, name='Cnt_Input')
    # c64
    d = Conv2D(64, (4, 4), strides=(2,2), padding='same', name='CntEnc1_Conv')(img_in)
    d = LeakyReLU(alpha=0.2, name='CntEnc1_relu')(d)
    # c128
    d = Conv2D(128, (4, 4), strides=(2,2), padding='same', name='CntEnc2_Conv')(d)
    d = BatchNormalization(name='CntEnc2_norm')(d, training=True)
    d = LeakyReLU(alpha=0.2, name='CntEnc2_relu')(d)
    # c256
    d = Conv2D(256, (4, 4), strides=(2,2), padding='same', name='CntEnc3_Conv')(d)
    d = BatchNormalization(name='CntEnc3_norm')(d, training=True)
    d = LeakyReLU(alpha=0.2, name='CntEnc3_relu')(d)
    # c512
    d = Conv2D(512, (4, 4), strides=(2,2), padding='same', name='CntEnc4_Conv')(d)
    d = BatchNormalization(name='CntEnc4_norm')(d, training=True)
    d = LeakyReLU(alpha=0.2, name='CntEnc4_relu')(d)
    # c512
    d = Conv2D(512, (4, 4), strides=(2,2), padding='same', name='CntEnc5_Conv')(d)
    d = BatchNormalization(name='CntEnc5_norm')(d, training=True)
    d = LeakyReLU(alpha=0.2, name='CntEnc5_relu')(d)
    # c-latent size
    d = Conv2D(latent_size, (4, 4), strides=(2,2), padding='same', name='CntEnc6_Conv')(d)
    d = BatchNormalization(name='CntEnc6_norm')(d, training=True)
    d = LeakyReLU(alpha=0.2, name='CntEnc6_relu')(d)
    #output
    output = GlobalMaxPool2D(name='output')(d)

    #define model
    model = Model(inputs=img_in, outputs=output, name='content_base_encoder')
    return model

class ContentNet(tf.keras.Model):

    def __init__(self, base_model):
        super(ContentNet, self).__init__()
        self._model = base_model 

    @tf.function
    def call(self, inputs):
        cnt_img, trans_img = inputs
        with tf.name_scope("Content") as scope:
            ft1 = self._model(cnt_img)

        with tf.name_scope("Transfer") as scope:
            ft2 = self._model(trans_img)

        return [ft1, ft2]


gc_model = define_cnt_encoder(64)
# CntNet = ContentNet(gc_model)
tf.keras.utils.plot_model(gc_model, show_shapes=True)

#%%    