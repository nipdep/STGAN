#%%
import tensorflow as tf  
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal, HeUniform
from tensorflow.keras.models import Model
from tensorflow_addons.layers import InstanceNormalization
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Reshape, Conv2DTranspose, LeakyReLU, Activation, Dropout, BatchNormalization, LeakyReLU, GlobalMaxPool2D, Concatenate, ReLU, AveragePooling2D
from tensorflow.keras import losses
from tensorflow.keras import metrics
from tensorflow.python.ops.gen_array_ops import Reshape
#import config 
# import cntdesc_model,stldesc_model
# from model.cntdesc_model import ContentNet

#%%

def define_generator(cnt_model, stl_model):

    for layer in cnt_model.layers:
        layer.trainable = False
    
    for layer in stl_model.layers:
        layer.trainable = False

    init = RandomNormal(stddev=0.02)

    cnt_vec = cnt_model.output
    stl_vec = stl_model.get_layer("latent_layer").output

    #full_latent_vec = Concatenate(name='Combined_latent_layer')([cnt_vec, stl_vec])
    full_latent_vec = stl_vec
    latent_mat = tf.keras.layers.Reshape((1, 1, 32))(full_latent_vec)
    #decoder model with skip connections
    #size:2
    g = Conv2DTranspose(256, (4, 4), strides=(2,2), padding='same', kernel_initializer=init, name='dec1_ConvT')(latent_mat)
    g = InstanceNormalization(name='dec1_norm')(g, training=True)
    g = LeakyReLU(alpha=0.2,name='dec1_relu')(g)
    #size:4
    g = Conv2DTranspose(256, (4, 4), strides=(2,2), padding='same', kernel_initializer=init, name='dec2_ConvT')(g)
    g = InstanceNormalization(name='dec2_norm')(g, training=True)
    g = Dropout(0.4, name='dec2_dropout')(g, training=True)
    g = Concatenate(name='dec2_concat')([g, cnt_model.get_layer('CntEnc5_relu').output])
    g = LeakyReLU(alpha=0.2,name='dec2_relu')(g)
    #size:8
    g = Conv2DTranspose(256, (4, 4), strides=(2,2), padding='same', kernel_initializer=init, name='dec3_ConvT')(g)
    g = InstanceNormalization(name='dec3_norm')(g, training=True)
    g = Dropout(0.4, name='dec3_dropout')(g, training=True)
    g = Concatenate(name='dec3_concat')([g, cnt_model.get_layer('CntEnc4_relu').output])
    g = LeakyReLU(alpha=0.2,name='dec3_relu')(g)
    #size:16
    g = Conv2DTranspose(128, (4, 4), strides=(2,2), padding='same', kernel_initializer=init, name='dec4_ConvT')(g)
    g = InstanceNormalization(name='dec4_norm')(g, training=True)
    g = Concatenate(name='dec4_concat')([g, cnt_model.get_layer('CntEnc3_relu').output])
    g = LeakyReLU(alpha=0.2,name='dec4_relu')(g)
    #size:32
    g = Conv2DTranspose(64, (4, 4), strides=(2,2), padding='same', kernel_initializer=init, name='dec5_ConvT')(g)
    g = InstanceNormalization(name='dec5_norm')(g, training=True)
    g = Concatenate(name='dec5_concat')([g, cnt_model.get_layer('CntEnc2_relu').output])
    g = LeakyReLU(alpha=0.2,name='dec5_relu')(g)
    #size:64
    g = Conv2DTranspose(32, (4, 4), strides=(2,2), padding='same', kernel_initializer=init, name='dec6_ConvT')(g)
    g = InstanceNormalization(name='dec6_norm')(g, training=True)
    g = Concatenate(name='dec6_concat')([g, cnt_model.get_layer('CntEnc1_relu').output])
    g = LeakyReLU(alpha=0.2, name='dec6_relu')(g)
    #size:128
    g = Conv2DTranspose(3, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init, name='dec7_ConvT')(g)
    #output layer
    output = Activation('tanh')(g)

    model = Model(inputs=[cnt_model.input, stl_model.input], outputs=output, name='Image_Generator')
    return model

# gc_model = cntdesc_model.define_cnt_encoder(32, (128, 128, 3))
# gs_model = stldesc_model.stl_encoder(32, (128, 128, 3))
# gen_model = define_generator(gc_model, gs_model)
# ds_model = stldesc_model.StyleNet(gs_model)
# dc_model = cntdesc_model.ContentNet(gc_model)
# gen_model.summary()
# tf.keras.utils.plot_model(gen_model, show_shapes=True)

#%%

def define_gan(g_model, dc_model, ds_model, image_shape=(128, 128, 3)):
    for layer in dc_model.layers:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = False
    
    for layer in ds_model.layers:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = True
    # input layer for GAN model
    cnt_img = Input(shape=image_shape)
    style_img = Input(shape=image_shape)
    # generator model
    gen_out = g_model([cnt_img, style_img])
    # style descriminator model
    
    dss_out, dst_out = ds_model([style_img, gen_out])
    # content descriminator model
    cnt_out, dct_out = dc_model([cnt_img, gen_out])
    model = Model(inputs=[cnt_img, style_img], outputs=[gen_out, dss_out, dst_out, cnt_out, dct_out])
    return model

# gan_model = define_gan(gen_model, dc_model, ds_model)
# tf.keras.utils.plot_model(gan_model, show_shapes=True)

# %%
