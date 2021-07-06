#%%
import tensorflow as tf  
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal, HeUniform
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Reshape, Conv2DTranspose, LeakyReLU, Activation, Dropout, BatchNormalization, LeakyReLU, GlobalMaxPool2D, Concatenate, ReLU, AveragePooling2D
from tensorflow.keras import losses
from tensorflow.keras import metrics
from tensorflow.python.ops.gen_array_ops import Reshape
#import config 
import cntdesc_model,stldesc_model

#%%

def define_generator(cnt_model, stl_model, image_shape):

    cnt_image = Input(image_shape)
    stl_image = Input(image_shape)
    init = RandomNormal(stddev=0.02)

    cnt_vec = cnt_model.output
    stl_vec = stl_model.output

    full_latent_vec = Concatenate(name='Combined_latent_layer')([cnt_vec, stl_vec])
    latent_mat = tf.keras.layers.Reshape((1, 1, 128))(full_latent_vec)
    #decoder model with skip connections
    #size:2
    g = Conv2DTranspose(512, (4, 4), strides=(2,2), padding='same', kernel_initializer=init, name='dec1_ConvT')(latent_mat)
    g = BatchNormalization(name='dec1_norm')(g, training=True)
    g = ReLU(name='dec1_relu')(g)
    #size:4
    g = Conv2DTranspose(512, (4, 4), strides=(2,2), padding='same', kernel_initializer=init, name='dec2_ConvT')(g)
    g = BatchNormalization(name='dec2_norm')(g, training=True)
    g = Dropout(0.4, name='dec2_dropout')(g, training=True)
    g = Concatenate(name='dec2_concat')([g, cnt_model.get_layer('CntEnc5_relu').output])
    g = ReLU(name='dec2_relu')(g)
    #size:8
    g = Conv2DTranspose(512, (4, 4), strides=(2,2), padding='same', kernel_initializer=init, name='dec3_ConvT')(g)
    g = BatchNormalization(name='dec3_norm')(g, training=True)
    g = Dropout(0.4, name='dec3_dropout')(g, training=True)
    g = Concatenate(name='dec3_concat')([g, cnt_model.get_layer('CntEnc4_relu').output])
    g = ReLU(name='dec3_relu')(g)
    #size:16
    g = Conv2DTranspose(256, (4, 4), strides=(2,2), padding='same', kernel_initializer=init, name='dec4_ConvT')(g)
    g = BatchNormalization(name='dec4_norm')(g, training=True)
    g = Concatenate(name='dec4_concat')([g, cnt_model.get_layer('CntEnc3_relu').output])
    g = ReLU(name='dec4_relu')(g)
    #size:32
    g = Conv2DTranspose(128, (4, 4), strides=(2,2), padding='same', kernel_initializer=init, name='dec5_ConvT')(g)
    g = BatchNormalization(name='dec5_norm')(g, training=True)
    g = Concatenate(name='dec5_concat')([g, cnt_model.get_layer('CntEnc2_relu').output])
    g = ReLU(name='dec5_relu')(g)
    #size:64
    g = Conv2DTranspose(64, (4, 4), strides=(2,2), padding='same', kernel_initializer=init, name='dec6_ConvT')(g)
    g = BatchNormalization(name='dec6_norm')(g, training=True)
    g = Concatenate(name='dec6_concat')([g, cnt_model.get_layer('CntEnc1_relu').output])
    g = ReLU(name='dec6_relu')(g)
    #size:128
    g = Conv2DTranspose(3, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init, name='dec7_ConvT')(g)
    #output layer
    output = Activation('tanh')(g)

    model = Model(inputs=[cnt_model.input, stl_model.input], outputs=output, name='Image_Generator')
    return model

gc_model = cntdesc_model.define_cnt_encoder(64)
gs_model = stldesc_model.define_desc_encoder(64)
gen_model = define_generator(gc_model, gs_model, (128, 128, 3))
tf.keras.utils.plot_model(gen_model, show_shapes=True)

#%%