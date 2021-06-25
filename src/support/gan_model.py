# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import time
import random
import tensorflow as tf
import numpy as np
from numpy import load
from numpy import zeros
from numpy import ones
from numpy.random import randint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Conv2DTranspose, LeakyReLU, Activation, Dropout, BatchNormalization, ReLU, LeakyReLU, Concatenate
from tensorflow.keras import losses
from tensorflow.keras import metrics 
from matplotlib import pyplot
from tensorflow.python.autograph.pyct import transformer


# %%
def define_encoder_block(layer_in, n_filters, batchnorm=True):
    init = RandomNormal(stddev=0.02)
    g = Conv2D(n_filters, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(layer_in)
    if batchnorm:
        g =  BatchNormalization()(g, training=True)
    g = LeakyReLU(alpha=0.2)(g)
    return g

def define_decoder_block(layer_in, skip_in, n_filters, dropout=True):
    init = RandomNormal(stddev=0.02)
    g = Conv2DTranspose(n_filters, (4, 4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
    g = BatchNormalization()(g, training=True)
    if dropout:
        g = Dropout(0.4)(g, training=True)
    g = Concatenate()([g, skip_in])
    g = ReLU()(g)
    return g


# %%
def defing_generator(image_shape=(128, 128, 3)):
    init = RandomNormal(stddev=0.02)
    content_image = Input(shape=image_shape)
    style_image = Input(shape=image_shape)
    # stack content and style images
    stacked_layer = Concatenate()([content_image, style_image])
    #encoder model
    e1 = define_encoder_block(stacked_layer, 64, batchnorm=False)
    e2 = define_encoder_block(e1, 128)
    e3 = define_encoder_block(e2, 256)
    e4 = define_encoder_block(e3, 512)
    e5 = define_encoder_block(e4, 512)
    e6 = define_encoder_block(e5, 512)
    #e7 = define_encoder_block(e6, 512)
    # bottleneck layer
    b = Conv2D(512, (4, 4), strides=(2, 2), padding='same', kernel_constraint=init)(e6)
    b = ReLU()(b)
    #decoder model
    #d1 = define_decoder_block(b, e7, 512)
    d2 = define_decoder_block(b, e6, 512)
    d3 = define_decoder_block(d2, e5, 512)
    d4 = define_decoder_block(d3, e4, 512, dropout=False)
    d5 = define_decoder_block(d4, e3, 256, dropout=False)
    d6 = define_decoder_block(d5, e2, 128, dropout=False)
    d7 = define_decoder_block(d6, e1, 64, dropout=False)
    #ouutput layer
    g = Conv2DTranspose(3, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d7)
    out_image = Activation('tanh')(g)
    model = Model(inputs=[content_image, style_image], outputs=out_image)
    return model

#%%

g_model = defing_generator()
tf.keras.utils.plot_model(g_model, show_shapes=True)

# %%
def define_cnt_descriminator(image_shape=(128, 128, 3)):
    init = RandomNormal(stddev=0.02)
    #content image input
    in_cnt_image = Input(shape=image_shape)
    #transfer image input 
    in_tr_image = Input(shape=image_shape)
    #concatnate image channel-wise
    merged = Concatenate()([in_cnt_image, in_tr_image])
    # c64
    d = Conv2D(64, (4, 4), strides=(2,2), padding='same', kernel_initializer=init)(merged)
    d = LeakyReLU(alpha=0.2)(d)
    # c128
    d = Conv2D(128, (4, 4), strides=(2,2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # c256
    d = Conv2D(256, (4, 4), strides=(2,2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # c512
    d = Conv2D(512, (4, 4), strides=(2,2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # patch output
    d = Conv2D(512, (4, 4), strides=(2,2), padding='same', kernel_initializer=init)(d)
    patch_out = Activation('sigmoid')(d)
    #define model
    model = Model(inputs=[in_cnt_image, in_tr_image], outputs=patch_out)
    return model

dsc_model = define_cnt_descriminator()
tf.keras.utils.plot_model(dsc_model, show_shapes=True)

# %%
def define_style_descrminator(image_size=(128, 128, 3)):
    init = RandomNormal(stddev=0.02)
    input_img = Input(shape=image_size)
    # C64
    d = Conv2D(64, (4, 4), (4, 4), padding='SAME', kernel_initializer=init)(input_img)
    d = LeakyReLU(alpha=0.2)(d)
	# C128
    d = Conv2D(128, (4, 4), (4, 4), padding='SAME', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
	# C256
    d = Conv2D(256, (4, 4), (4, 4), padding='SAME', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # flatten
    flt = Flatten()(d)
    # linear logits layer
    output = Dense(1)(flt)
    #build and compile the model
    model = Model(inputs=input_img, outputs=output, name='style_descriminator')
    return model

dss_model = define_style_descrminator()
tf.keras.utils.plot_model(dss_model, show_shapes=True)

# %%
def define_gan(g_model, dc_model, ds_model, image_shape=(128, 128, 3)):
    for layer in dc_model.layers:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = False
    
    for layer in ds_model.layers:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = False
    # input layer for GAN model
    cnt_img = Input(shape=image_shape)
    style_img = Input(shape=image_shape)
    # generator model
    gen_out = g_model([cnt_img, style_img])
    # style descriminator model
    dss_out = ds_model(style_img)
    dst_out = ds_model(gen_out)
    # content descriminator model
    cnt_out = dc_model([cnt_img, gen_out])
    model = Model(inputs=[cnt_img, style_img], outputs=[gen_out,  dss_out, dst_out, cnt_out])
    return model

gan_model = define_gan(g_model, dsc_model, dss_model)
tf.keras.utils.plot_model(gan_model, show_shapes=True)

#%%

def pairWiseRankingLoss(y_ref, y_style, label):
    m  = tf.cast(tf.broadcast_to(1, shape=y_ref.shape), dtype=tf.float32)
    u  = tf.cast(tf.broadcast_to(0, shape=y_ref.shape), dtype=tf.float32)
    i  = tf.cast(tf.broadcast_to(1, shape=y_ref.shape), dtype=tf.float32)
    y = tf.cast(label[..., tf.newaxis], dtype=tf.float32)
    dist = tf.norm(y_ref-y_style, ord='euclidean', axis=-1, keepdims=True)
    loss = y*dist + (i-y)*tf.reduce_max(tf.stack([u,m-dist]), axis=0)
    return tf.reduce_mean(loss)
dscLoss = tf.keras.losses.binary_crossentropy(from_logits=True)
cntLoss = tf.keras.losses.KLDivergence()

gan_opt = tf.keras.optimizers.Adamax(lr=0.002)

gan_alpha = 0.6
gan_beta = 0.5


#%%
@tf.function
def gan_train_step(ref_in, style_in, trans_in,cnt_true, style_true):
    with tf.GradientTape() as tape:
        gen_out, dss_out, dst_out, cnt_out = gan_model(ref_in, style_in)
        dss_loss = pairWiseRankingLoss(dss_out, dst_out, style_true)
        dsc_loss = dscLoss(cnt_out, cnt_true)
        gen_loss = cntLoss(trans_in, gen_out)
    total_loss =  gan_alpha*(gan_beta*dss_loss+(1-gan_beta)*dsc_loss)+(1-gan_alpha)*gen_loss
    grads = tape.gradient(total_loss, gan_model)
    gan_opt.apply_gradients(zip(grads, gan_model.trainable_weights))
    return total_loss, dss_loss, dsc_loss

#%%

ds_model = define_style_descrminator() #TODO
ds_opt = tf.keras.optimizers.Adam(lr=0.02)

#%%

@tf.function
def ds_train_step(style_in, trans_in, label_in):
    with tf.GradientTape() as tape:
        ref_out = ds_model(style_in)
        trans_out = ds_model(trans_in)
        loss = pairWiseRankingLoss(ref_out, trans_out, label_in)
    grads = tape.gradient(loss, ds_model.trainable_weights)
    ds_opt.apply_gradients(zip(grads, ds_model.trainable_weights))
    #train_metrics.update_state(ref_out, style_out, label_in)
    return loss

#%%

dc_model = define_cnt_descriminator()
dc_opt = tf.keras.optimizers.Adam(lr=0.02)


#%%

@tf.function
def dc_train_step(cnt_in, trans_in, label_in):
    with tf.GradientTape() as tape:
        logits = dc_model(cnt_in, trans_in)
        loss = pairWiseRankingLoss(label_in, logits)
    grads = tape.gradient(loss, dc_model.trainable_weights)
    dc_opt.apply_gradients(zip(grads, dc_model.trainable_weights))
    #train_metrics.update_state(ref_out, style_out, label_in)
    return loss

#%%

def load_pixel_metrics(filename):
    full_mat = np.load(filename)
    style_pixels = (full_mat['style']-127.5)/127.5
    content_pixels = (full_mat['cotent']-127.5)/127.5
    transfer_mat  = (full_mat['transfers']-127.5)/127.5
    return style_pixels, content_pixels, transfer_mat

def generate_real_samples(dataset, n_samples, patch_shape):
    style, content, trans = dataset
    cnt_idxs = random.sample(range(style.shape[1]), n_samples)
    style_idxs = np.random.randint(0, style.shape[0], n_samples)

    cnt_pixels = content[cnt_idxs]
    style_pixels = style[style_idxs]
    mat_pixels = trans[style_idxs, cnt_idxs, ...]

    y_dc = ones((n_samples, patch_shape, patch_shape, 1))
    y_ds = ones((n_samples))
    return [cnt_pixels, style_pixels, mat_pixels], y_dc, y_ds

def generate_fake_samples(g_model, samples, patch_shape):
    cnt_img, style_img = samples
    X = g_model.predict(cnt_img, style_img)
    y_dc = zeros((len(X), patch_shape, patch_shape, 1))
    y_ds = zeros((len(X)))
    return X, y_dc, y_ds

#%%

def train(g_model, ds_model, dc_model, gan_model, dataset, n_epoch=100, batch_size=2):
    n_patch = dc_model.output_shape[1]
    batch_per_epoch = len(dataset[1])//batch_size
    n_steps = n_epoch*batch_per_epoch
    for i in range(n_steps):
        [X_cnt, X_stl, X_trn], ydc_real, yds_real = generate_real_samples(dataset, batch_size, n_patch)
        X_fake_trn, ydc_fake, yds_fake = generate_fake_samples(g_model, [X_cnt, X_stl], n_patch)
        # train style descriminator
        ds_loss1 = ds_train_step(X_stl, X_trn, yds_real)
        ds_loss2 = ds_train_step(X_stl, X_fake_trn, yds_fake)
        #train content descriminator
        dc_loss1 = dc_train_step(X_cnt, X_trn, ydc_real)
        dc_loss2 = dc_train_step(X_cnt, X_fake_trn, ydc_fake)
        #train GAN model
        gan_total_loss, gan_dss_loss, gan_dsc_loss = gan_train_step(X_cnt, X_stl, X_fake_trn, ydc_real, yds_real)


#%%


