
#%%
import tensorflow as tf
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Model
import tensorflow_addons as tfa
import tensorflow.keras.backend as K
from tensorflow_addons.layers import InstanceNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal, HeUniform
from tensorflow.keras.layers import Input, Conv2D, Subtract, Flatten, Dense, Conv2DTranspose, LeakyReLU, Activation, Dropout, BatchNormalization, LeakyReLU, GlobalMaxPool2D, Concatenate, ReLU, AveragePooling2D, Lambda, Reshape, Add, Reshape
from tensorflow.keras import losses
from tensorflow.keras import metrics
from tensorflow.python.keras.layers.pooling import GlobalAveragePooling2D
#import config
#%%

def WaveletTransformAxisY(batch_img):
    odd_img  = batch_img[:,0::2]
    even_img = batch_img[:,1::2]
    L = (odd_img + even_img) / 2.0
    H = K.abs(odd_img - even_img)
    return L, H

def WaveletTransformAxisX(batch_img):
    # transpose + fliplr
    tmp_batch = K.permute_dimensions(batch_img, [0, 2, 1])[:,:,::-1]
    _dst_L, _dst_H = WaveletTransformAxisY(tmp_batch)
    # transpose + flipud
    dst_L = K.permute_dimensions(_dst_L, [0, 2, 1])[:,::-1,...]
    dst_H = K.permute_dimensions(_dst_H, [0, 2, 1])[:,::-1,...]
    return dst_L, dst_H

def Wavelet(batch_image):
    # make channel first image
    batch_image = K.permute_dimensions(batch_image, [0, 3, 1, 2])
    r = batch_image[:,0]
    g = batch_image[:,1]
    b = batch_image[:,2]

    # level 1 decomposition
    wavelet_L, wavelet_H = WaveletTransformAxisY(r)
    r_wavelet_LL, r_wavelet_LH = WaveletTransformAxisX(wavelet_L)
    r_wavelet_HL, r_wavelet_HH = WaveletTransformAxisX(wavelet_H)

    wavelet_L, wavelet_H = WaveletTransformAxisY(g)
    g_wavelet_LL, g_wavelet_LH = WaveletTransformAxisX(wavelet_L)
    g_wavelet_HL, g_wavelet_HH = WaveletTransformAxisX(wavelet_H)

    wavelet_L, wavelet_H = WaveletTransformAxisY(b)
    b_wavelet_LL, b_wavelet_LH = WaveletTransformAxisX(wavelet_L)
    b_wavelet_HL, b_wavelet_HH = WaveletTransformAxisX(wavelet_H)

    wavelet_data = [r_wavelet_LL, r_wavelet_LH, r_wavelet_HL, r_wavelet_HH, 
                    g_wavelet_LL, g_wavelet_LH, g_wavelet_HL, g_wavelet_HH,
                    b_wavelet_LL, b_wavelet_LH, b_wavelet_HL, b_wavelet_HH]
    transform_batch = K.stack(wavelet_data, axis=1)

    # level 2 decomposition
    wavelet_L2, wavelet_H2 = WaveletTransformAxisY(r_wavelet_LL)
    r_wavelet_LL2, r_wavelet_LH2 = WaveletTransformAxisX(wavelet_L2)
    r_wavelet_HL2, r_wavelet_HH2 = WaveletTransformAxisX(wavelet_H2)

    wavelet_L2, wavelet_H2 = WaveletTransformAxisY(g_wavelet_LL)
    g_wavelet_LL2, g_wavelet_LH2 = WaveletTransformAxisX(wavelet_L2)
    g_wavelet_HL2, g_wavelet_HH2 = WaveletTransformAxisX(wavelet_H2)

    wavelet_L2, wavelet_H2 = WaveletTransformAxisY(b_wavelet_LL)
    b_wavelet_LL2, b_wavelet_LH2 = WaveletTransformAxisX(wavelet_L2)
    b_wavelet_HL2, b_wavelet_HH2 = WaveletTransformAxisX(wavelet_H2)


    wavelet_data_l2 = [r_wavelet_LL2, r_wavelet_LH2, r_wavelet_HL2, r_wavelet_HH2, 
                    g_wavelet_LL2, g_wavelet_LH2, g_wavelet_HL2, g_wavelet_HH2,
                    b_wavelet_LL2, b_wavelet_LH2, b_wavelet_HL2, b_wavelet_HH2]
    transform_batch_l2 = K.stack(wavelet_data_l2, axis=1)

    # level 3 decomposition
    wavelet_L3, wavelet_H3 = WaveletTransformAxisY(r_wavelet_LL2)
    r_wavelet_LL3, r_wavelet_LH3 = WaveletTransformAxisX(wavelet_L3)
    r_wavelet_HL3, r_wavelet_HH3 = WaveletTransformAxisX(wavelet_H3)

    wavelet_L3, wavelet_H3 = WaveletTransformAxisY(g_wavelet_LL2)
    g_wavelet_LL3, g_wavelet_LH3 = WaveletTransformAxisX(wavelet_L3)
    g_wavelet_HL3, g_wavelet_HH3 = WaveletTransformAxisX(wavelet_H3)

    wavelet_L3, wavelet_H3 = WaveletTransformAxisY(b_wavelet_LL2)
    b_wavelet_LL3, b_wavelet_LH3 = WaveletTransformAxisX(wavelet_L3)
    b_wavelet_HL3, b_wavelet_HH3 = WaveletTransformAxisX(wavelet_H3)

    wavelet_data_l3 = [r_wavelet_LL3, r_wavelet_LH3, r_wavelet_HL3, r_wavelet_HH3, 
                    g_wavelet_LL3, g_wavelet_LH3, g_wavelet_HL3, g_wavelet_HH3,
                    b_wavelet_LL3, b_wavelet_LH3, b_wavelet_HL3, b_wavelet_HH3]
    transform_batch_l3 = K.stack(wavelet_data_l3, axis=1)

    # level 4 decomposition
    wavelet_L4, wavelet_H4 = WaveletTransformAxisY(r_wavelet_LL3)
    r_wavelet_LL4, r_wavelet_LH4 = WaveletTransformAxisX(wavelet_L4)
    r_wavelet_HL4, r_wavelet_HH4 = WaveletTransformAxisX(wavelet_H4)

    wavelet_L4, wavelet_H4 = WaveletTransformAxisY(g_wavelet_LL3)
    g_wavelet_LL4, g_wavelet_LH4 = WaveletTransformAxisX(wavelet_L4)
    g_wavelet_HL4, g_wavelet_HH4 = WaveletTransformAxisX(wavelet_H4)

    wavelet_L3, wavelet_H3 = WaveletTransformAxisY(b_wavelet_LL3)
    b_wavelet_LL4, b_wavelet_LH4 = WaveletTransformAxisX(wavelet_L4)
    b_wavelet_HL4, b_wavelet_HH4 = WaveletTransformAxisX(wavelet_H4)


    wavelet_data_l4 = [r_wavelet_LL4, r_wavelet_LH4, r_wavelet_HL4, r_wavelet_HH4, 
                    g_wavelet_LL4, g_wavelet_LH4, g_wavelet_HL4, g_wavelet_HH4,
                    b_wavelet_LL4, b_wavelet_LH4, b_wavelet_HL4, b_wavelet_HH4]
    transform_batch_l4 = K.stack(wavelet_data_l4, axis=1)

    # print('shape before')
    # print(transform_batch.shape)
    # print(transform_batch_l2.shape)
    # print(transform_batch_l3.shape)
    # print(transform_batch_l4.shape)

    decom_level_1 = K.permute_dimensions(transform_batch, [0, 2, 3, 1])
    decom_level_2 = K.permute_dimensions(transform_batch_l2, [0, 2, 3, 1])
    decom_level_3 = K.permute_dimensions(transform_batch_l3, [0, 2, 3, 1])
    decom_level_4 = K.permute_dimensions(transform_batch_l4, [0, 2, 3, 1])
    
    # print('shape after')
    # print(decom_level_1.shape)
    # print(decom_level_2.shape)
    # print(decom_level_3.shape)
    # print(decom_level_4.shape)
    return [decom_level_1, decom_level_2, decom_level_3, decom_level_4]


def Wavelet_out_shape(input_shapes):
    # print('in to shape')
    return [tuple([None, 112, 112, 12]), tuple([None, 56, 56, 12]), 
            tuple([None, 28, 28, 12]), tuple([None, 14, 14, 12])]

def define_desc_encoder(latent_size, image_shape=(128, 128, 3)):
    style_image = Input(shape=image_shape, name='style_image')
    # wavelet transform style_image
    wavelet = Lambda(Wavelet, Wavelet_out_shape, name='wavelet')
    input_l1, input_l2, input_l3, input_l4 = wavelet(style_image)
    # print(input_l1)
    # print(input_l2)
    # print(input_l3)
    # print(input_l4)

    ## wavelet trasform style extraction head
    # level one decomposition starts
    conv_1 = Conv2D(64, kernel_size=(3, 3), padding='same', name='conv_1')(input_l1)
    norm_1 = InstanceNormalization(name='norm_1')(conv_1)
    relu_1 = Activation('relu', name='relu_1')(norm_1)

    conv_1_2 = Conv2D(64, kernel_size=(3, 3), strides=(2, 2), padding='same', name='conv_1_2')(relu_1)
    norm_1_2 = InstanceNormalization(name='norm_1_2')(conv_1_2)
    relu_1_2 = Activation('relu', name='relu_1_2')(norm_1_2)

    # level two decomposition starts
    conv_a = Conv2D(filters=64, kernel_size=(3, 3), padding='same', name='conv_a')(input_l2)
    norm_a = InstanceNormalization(name='norm_a')(conv_a)
    relu_a = Activation('relu', name='relu_a')(norm_a)

    # concate level one and level two decomposition
    concate_level_2 = Concatenate()([relu_1_2, relu_a])
    conv_2 = Conv2D(128, kernel_size=(3, 3), padding='same', name='conv_2')(concate_level_2)
    norm_2 = InstanceNormalization(name='norm_2')(conv_2)
    relu_2 = Activation('relu', name='relu_2')(norm_2)

    conv_2_2 = Conv2D(128, kernel_size=(3, 3), strides=(2, 2), padding='same', name='conv_2_2')(relu_2)
    norm_2_2 = InstanceNormalization(name='norm_2_2')(conv_2_2)
    relu_2_2 = Activation('relu', name='relu_2_2')(norm_2_2)

    # level three decomposition starts 
    conv_b = Conv2D(filters=64, kernel_size=(3, 3), padding='same', name='conv_b')(input_l3)
    norm_b = InstanceNormalization(name='norm_b')(conv_b)
    relu_b = Activation('relu', name='relu_b')(norm_b)

    conv_b_2 = Conv2D(128, kernel_size=(3, 3), padding='same', name='conv_b_2')(relu_b)
    norm_b_2 = InstanceNormalization(name='norm_b_2')(conv_b_2)
    relu_b_2 = Activation('relu', name='relu_b_2')(norm_b_2)

    # concate level two and level three decomposition 
    concate_level_3 = Concatenate()([relu_2_2, relu_b_2])
    conv_3 = Conv2D(256, kernel_size=(3, 3), padding='same', name='conv_3')(concate_level_3)
    norm_3 = InstanceNormalization(name='norm_3')(conv_3)
    relu_3 = Activation('relu', name='relu_3')(norm_3)

    conv_3_2 = Conv2D(256, kernel_size=(3, 3), strides=(2, 2), padding='same', name='conv_3_2')(relu_3)
    norm_3_2 = InstanceNormalization(name='norm_3_2')(conv_3_2)
    relu_3_2 = Activation('relu', name='relu_3_2')(norm_3_2)

    # level four decomposition start
    conv_c = Conv2D(64, kernel_size=(3, 3), padding='same', name='conv_c')(input_l4)
    norm_c = InstanceNormalization(name='norm_c')(conv_c)
    relu_c = Activation('relu', name='relu_c')(norm_c)

    conv_c_2 = Conv2D(256, kernel_size=(3, 3), padding='same', name='conv_c_2')(relu_c)
    norm_c_2 = InstanceNormalization(name='norm_c_2')(conv_c_2)
    relu_c_2 = Activation('relu', name='relu_c_2')(norm_c_2)

    conv_c_3 = Conv2D(256, kernel_size=(3, 3), padding='same', name='conv_c_3')(relu_c_2)
    norm_c_3 = InstanceNormalization(name='norm_c_3')(conv_c_3)
    relu_c_3 = Activation('relu', name='relu_c_3')(norm_c_3)

    # concate level level three and level four decomposition
    concate_level_4 = Concatenate()([relu_3_2, relu_c_3])
    conv_4 = Conv2D(256, kernel_size=(3, 3), padding='same', name='conv_4')(concate_level_4)
    norm_4 = InstanceNormalization(name='norm_4')(conv_4)
    relu_4 = Activation('relu', name='relu_4')(norm_4)

    conv_4_2 = Conv2D(256, kernel_size=(3, 3), strides=(2, 2), padding='same', name='conv_4_2')(relu_4)
    norm_4_2 = InstanceNormalization(name='norm_4_2')(conv_4_2)
    relu_4_2 = Activation('relu', name='relu_4_2')(norm_4_2)

    conv_5_1 = Conv2D(256, kernel_size=(3, 3), padding='same', name='conv_5_1')(relu_4_2)
    norm_5_1 = InstanceNormalization(name='norm_5_1')(conv_5_1)
    relu_5_1 = Activation('relu', name='relu_5_1')(norm_5_1)

    conv_5_2 = Conv2D(latent_size, kernel_size=(3, 3), padding='same', name='conv_5_2')(relu_5_1)
    norm_5_2 = InstanceNormalization(name='norm_5_2')(conv_5_2)
    relu_5_2 = Activation('relu', name='relu_5_2')(norm_5_2)

    pool = GlobalMaxPool2D(name='pool_logits')(relu_5_2)
    logits = Lambda(lambda x:tf.math.l2_normalize(x, axis=1), name='l2_norm')(pool)
    model = Model(inputs=style_image, outputs=logits)
    return model

#%%
def define_stl_encoder(latent_size, image_shape=(128, 128, 3)):
    #content image input
    img_in = Input(shape=image_shape, name='Stl_Input')
    # c64
    d = Conv2D(64, (5, 5), strides=(2,2), padding='same', name='StlEnc1_Conv')(img_in)
    d = LeakyReLU(alpha=0.2, name='StlEnc1_relu')(d)
    # c128
    d = Conv2D(128, (5, 5), strides=(2,2), padding='same', name='StlEnc2_Conv')(d)
    d = BatchNormalization(name='StlEnc2_norm')(d, training=True)
    d = LeakyReLU(alpha=0.2, name='StlEnc2_relu')(d)
    # c256
    d = Conv2D(256, (3, 3), strides=(2,2), padding='same', name='StlEnc3_Conv')(d)
    d = BatchNormalization(name='StlEnc3_norm')(d, training=True)
    d = LeakyReLU(alpha=0.2, name='StlEnc3_relu')(d)
    # c512
    d = Conv2D(512, (3, 3), strides=(2,2), padding='same', name='StlEnc4_Conv')(d)
    d = BatchNormalization(name='StlEnc4_norm')(d, training=True)
    d = LeakyReLU(alpha=0.2, name='StlEnc4_relu')(d)
    # c512
    d = Conv2D(512, (3, 3), strides=(2,2), padding='same', name='StlEnc5_Conv')(d)
    d = BatchNormalization(name='StlEnc5_norm')(d, training=True)
    d = LeakyReLU(alpha=0.2, name='StlEnc5_relu')(d)
    # c-latent size
    d = Conv2D(latent_size, (3, 3), strides=(2,2), padding='same', name='StlEnc6_Conv')(d)
    d = BatchNormalization(name='StlEnc6_norm')(d, training=True)
    d = LeakyReLU(alpha=0.2, name='StlEnc6_relu')(d)
    #output
    pool = GlobalMaxPool2D(name='stldisc_output')(d)
    output = Lambda(lambda x:tf.math.l2_normalize(x, axis=1), name='l2_norm')(pool)
    #define model
    model = Model(inputs=img_in, outputs=output, name='style_base_encoder')
    return model

#%%

def define_stl_emb_encoder(latent_size, n_classes, image_shape):
    idx = Input(shape=(), name='index_input')
    emb_layer = tf.keras.layers.Embedding(n_classes, 32, name="embedding")
    feat_model = tf.keras.applications.ResNet101V2(include_top=False, input_shape=image_shape)
    t = False
    for layer in feat_model.layers:
        if layer.name == "conv5_block3_1_conv":
            t = True
        layer.trainable = t
    x = feat_model.output
    x = GlobalAveragePooling2D(name='pool')(x)
    x = Dropout(0.3, name='dropout')(x)
    x = Dense(latent_size, activation='relu')(x)
    enc_out = Lambda(lambda x:tf.math.l2_normalize(x, axis=1), name='StlL2_norm')(x)
    y = emb_layer(idx)
    dist = Subtract(name="subs")([enc_out, y])

    model = Model(inputs=[feat_model.input,idx], outputs=dist, name='style_base_encoder')
    return model
#%%
# gen_model = define_stl_encoder(32, 36, (128, 128, 3))
#gen_model.summary()
# tf.keras.utils.plot_model(gen_model, show_shapes=True)
#%%
class StyleNet(tf.keras.Model):

    def __init__(self, base_model):
        super(StyleNet, self).__init__()
        self._model = base_model 

    @tf.function
    def call(self, inputs):
        ref_img, trans_img = inputs
        with tf.name_scope("Style") as scope:
            ft1 = self._model(ref_img)
            #ft1 = tf.math.l2_normalize(ft1, axis=-1)
        with tf.name_scope("STransfer") as scope:
            ft2 = self._model(trans_img)
            #ft2 = tf.math.l2_normalize(ft2, axis=-1)
        return [ft1, ft2]
    
    @tf.function
    def get_features(self, inputs):
        return tf.math.l2_normalize(self._model(inputs), axis=-1)

# gs_model = define_desc_encoder(64)
# CntNet = StyleNet(gs_model)
# tf.keras.utils.plot_model(gs_model, show_shapes=True)

#base_model = define_style_descrminator(config.DESCS_LATENT_SIZE, config.IMAGE_SHAPE)

# %%

def define_stl_regressor(latent_size, image_size=(128, 128, 3)):
    feat_model = tf.keras.applications.densenet.DenseNet121(weights='imagenet', include_top=False, input_shape=image_size)
    t = True
    for layer in feat_model.layers:
        # if layer.name == "conv5_block16_0_bn":
        #     t = True 
        layer.trainable=t
    x = feat_model.output
    # x = Conv2D(1024, (3, 3), (2, 2), padding='same', name='stlreg_c512')(x)
    # x = Conv2D(latent_size, (3, 3), (2, 2), padding='same', name='stlreg_latent_ly')(x)
    # x = Conv2D(1, (3, 3), (2, 2), padding='same', name='stlreg_c1')(x)
    # output = Reshape((1,), name='stlreg_output')(x)
    x = GlobalMaxPool2D(name='pool')(x)
    #x = Dense(512, activation='relu')(x)
    #x = Dropout(rate=0.3, name='stl_dropout')(x)
    x = Dense(latent_size, activation='relu', name='latent_layer')(x)
    output = Dense(1, use_bias=False, name='output')(x)

    model = Model(inputs=feat_model.input, outputs=output, name='style_classifier')
    return model

# sr_model = define_stl_regressor(32)
# # CntNet = StyleNet(gs_model)
# tf.keras.utils.plot_model(sr_model, show_shapes=True)
#%%
def define_stl_classifier(latent_size, image_size=(128, 128, 3)):
    feat_model = tf.keras.applications.densenet.DenseNet121(weights='imagenet', include_top=False, input_shape=image_size)
    t = False
    for layer in feat_model.layers:
        # if layer.name == "conv5_block3_1_conv":
        #     t = True
        layer.trainable = True
    x = feat_model.output
    # x = Conv2D(1024, (3, 3), (2, 2), padding='same', name='stlreg_c512')(x)
    # x = Conv2D(latent_size, (3, 3), (2, 2), padding='same', name='stlreg_latent_ly')(x)
    # x = Conv2D(12, (3, 3), (2, 2), padding='same', name='stlreg_c1')(x)
    # x = Reshape((12,), name='stl_reshape')(x)
    # output = Activation('softmax', name='stlreg_output')(x)
    x = GlobalMaxPool2D(name='pool')(x)
    x = Dropout(rate=0.2, name='stl_dropout')(x)
    x = Dense(512, activation='relu')(x)
    #x = Dense(512, activation='relu')(x)
    x = Dense(latent_size, activation='relu', use_bias=True, name='latent_layer')(x)
    output = Dense(12, activation='softmax', name='output')(x)

    model = Model(inputs=feat_model.input, outputs=output, name='style_classifier')
    return model

#%%

# gen_model = define_stl_classifier(32, 36, (128, 128, 3))
# gen_model.summary()
#tf.keras.utils.plot_model(gen_model, show_shapes=True)
#%%

class EmbStyleNet(tf.keras.Model):

    def __init__(self, base_model):
        super(EmbStyleNet, self).__init__()
        self._model = base_model 

    @tf.function
    def call(self, inputs):
        ref_img, style_img, ref_lbl, style_lbl = inputs
        with tf.name_scope("RefT") as scope:
            ft1 = self._model([ref_img, ref_lbl])
            #ft1 = tf.math.l2_normalize(ft1, axis=-1)
        with tf.name_scope("RefF") as scope:
            ft2 = self._model([ref_img, style_lbl])
            #ft2 = tf.math.l2_normalize(ft2, axis=-1)
        with tf.name_scope("StyleT") as scope:
            ft3 = self._model([style_img, style_lbl])
            #ft1 = tf.math.l2_normalize(ft1, axis=-1)
        with tf.name_scope("StyleF") as scope:
            ft4 = self._model([style_img, ref_lbl])
            #ft2 = tf.math.l2_normalize(ft2, axis=-1)
        return [ft1, ft2, ft3, ft4]
    
    @tf.function
    def get_features(self, inputs):
        return tf.math.l2_normalize(self._model(inputs), axis=-1)

#%%
def stl_encoder(latent_size, image_size=(128, 128, 3)):
    feat_model = tf.keras.applications.densenet.DenseNet121(weights='imagenet', include_top=False, input_shape=image_size)
    # feat_model = tf.keras.applications.vgg16.VGG16(weights='imagenet', include_top=False, input_shape=image_size)

    t = False
    for layer in feat_model.layers:
        # if layer.name == "conv5_block16_0_bn":
        #     t = True 
        layer.trainable=t
    x = feat_model.output
    # x = Conv2D(1024, (3, 3), (2, 2), padding='same', name='stlreg_c512')(x)
    # x = Conv2D(latent_size, (3, 3), (2, 2), padding='same', name='stlreg_latent_ly')(x)
    # x = Conv2D(1, (3, 3), (2, 2), padding='same', name='stlreg_c1')(x)
    # output = Reshape((1,), name='stlreg_output')(x)
    x = GlobalAveragePooling2D(name='pool')(x)
    #x = Dense(512, activation='relu')(x)
    #x = Dropout(rate=0.3, name='stl_dropout')(x)
    x = Dense(latent_size, activation='relu', name='latent_layer')(x)
    output = Lambda(lambda x:tf.math.l2_normalize(x, axis=-1), name='StlL2_norm')(x)
    model = Model(inputs=feat_model.input, outputs=output, name='style_enc')
    return model

# sr_model = stl_encoder(32)
# # CntNet = StyleNet(gs_model)
# tf.keras.utils.plot_model(sr_model, show_shapes=True)
#%%


