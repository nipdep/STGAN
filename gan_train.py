##! usr/bin/python3
# %%

import config
from src.model.desc_model import define_style_descrminator, StyleNet
from src.model.gan_model import define_cnt_descriminator, define_gan, defing_generator

import os 
import logging
import time
import random
from datetime import datetime
from livelossplot import outputs
import tensorflow as tf
import numpy as np
from numpy import load, zeros, ones
from numpy.random import randint
from sklearn.utils import shuffle
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Conv2DTranspose, LeakyReLU, Activation, Dropout, BatchNormalization, ReLU, LeakyReLU, Concatenate, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from tensorflow.keras.initializers import HeUniform
from tensorflow.keras import losses
from tensorflow.keras import metrics 
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.math import multiply, add, subtract
from matplotlib import pyplot
from tensorflow.python.autograph.pyct import transformer
from livelossplot import PlotLosses
from livelossplot.outputs import MatplotlibPlot

#%%
# set logger
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

#tensorboard logger
logdir = config.LOG_DIR+ "/gan_" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=logdir, histogram_freq=1, profile_batch=1)

def pairWiseRankingLoss(y_ref, y_style, label):
    m  = tf.cast(tf.broadcast_to(1, shape=y_ref.shape), dtype=tf.float32)
    u  = tf.cast(tf.broadcast_to(0, shape=y_ref.shape), dtype=tf.float32)
    i  = tf.cast(tf.broadcast_to(1, shape=y_ref.shape), dtype=tf.float32)
    y = tf.cast(label[..., tf.newaxis], dtype=tf.float32)
    dist = tf.norm(y_ref-y_style, ord=1, axis=-1, keepdims=True)
    loss = y*dist + (i-y)*tf.reduce_max(tf.stack([u,m-dist]), axis=0)
    return tf.cast(tf.reduce_mean(loss), dtype=tf.float32)

def ganLoss(dss_loss, dsc_loss, gen_loss):
    gan_alpha = config.GAN_ALPHA
    gan_beta = config.GAN_BETA
    one = 1

    tot_loss = gan_alpha*(gan_beta*dss_loss+(one-gan_beta)*dsc_loss)+(one-gan_alpha)*gen_loss
    return tot_loss

dscLoss = tf.keras.losses.BinaryCrossentropy()
cntLoss = tf.keras.losses.MeanAbsoluteError()

gan_opt = tf.keras.optimizers.Adam(lr=0.002)


@tf.function
def gan_train_step(ref_in, style_in, trans_in, cnt_true, style_true):
    with tf.GradientTape() as tape:
        gen_out, dss_out, dst_out, cnt_out = gan_model([ref_in, style_in])
        dss_loss = pairWiseRankingLoss(dss_out, dst_out, style_true)
        dsc_loss = tf.cast(dscLoss(cnt_true, cnt_out), dtype=tf.float32)
        gen_loss = tf.cast(tf.math.abs(cntLoss(trans_in, gen_out)), dtype=tf.float32)
        total_loss = ganLoss(dss_loss, dsc_loss, gen_loss)
        
    #total_loss = add(add(multiply(gan_beta, dss_loss), multiply(subtract(one, gan_beta), dsc_loss)), multiply(subtract(one, gan_alpha), gen_loss))    
    #total_loss =  gan_alpha*(gan_beta*dss_loss+(one-gan_beta)*dsc_loss)+(one-gan_alpha)*gen_loss
    grads = tape.gradient(total_loss, gan_model.trainable_weights)
    gan_opt.apply_gradients(zip(grads, gan_model.trainable_weights))
    return total_loss, dss_loss, dsc_loss, gen_loss

ds_opt = tf.keras.optimizers.Adam(lr=0.02)

@tf.function
def ds_train_step(style_in, trans_in, label_in):
    with tf.GradientTape() as tape:
        ref_out, trans_out = ds_model([style_in, trans_in])
        loss = pairWiseRankingLoss(ref_out, trans_out, label_in)
    grads = tape.gradient(loss, ds_model.trainable_weights)
    ds_opt.apply_gradients(zip(grads, ds_model.trainable_weights))
    #train_metrics.update_state(ref_out, style_out, label_in)
    return loss


dc_opt = tf.keras.optimizers.Adam(lr=0.02)

@tf.function
def dc_train_step(cnt_in, trans_in, label_in):
    with tf.GradientTape() as tape:
        logits = dc_model([cnt_in, trans_in])
        loss = cntLoss(label_in, logits)
    grads = tape.gradient(loss, dc_model.trainable_weights)
    dc_opt.apply_gradients(zip(grads, dc_model.trainable_weights))
    #train_metrics.update_state(ref_out, style_out, label_in)
    return loss


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
    X = g_model([cnt_img, style_img])
    y_dc = zeros((len(X), patch_shape, patch_shape, 1))
    y_ds = zeros((len(X)))
    return X, y_dc, y_ds

def summarize_performance(step, g_model, dataset, n_samples=3):
    [X_cnt, X_stl, X_trn], _, _ = generate_real_samples(dataset, n_samples, 1)
    X_fake, _, _ = generate_fake_samples(g_model, [X_cnt, X_stl], 1)
    #rescale pixels values
    X_cnt = (X_cnt+1)/2.0
    X_stl = (X_stl+1)/2.0
    X_trn = (X_trn+1)/2.0
    X_fake = (X_fake+1)/2.0
    # plot samples
    for i in range(n_samples):
        pyplot.subplot(4, n_samples, 1 + i)
        pyplot.axis('off')
        pyplot.imshow(X_cnt[i])
    for i in range(n_samples):
        pyplot.subplot(4, n_samples, 1 + n_samples + i)
        pyplot.axis('off')
        pyplot.imshow(X_stl[i])
    for i in range(n_samples):
        pyplot.subplot(4, n_samples, 1 + 2*n_samples + i)
        pyplot.axis('off')
        pyplot.imshow(X_trn[i])
    for i in range(n_samples):
        pyplot.subplot(4, n_samples, 1 + 3*n_samples + i)
        pyplot.axis('off')
        pyplot.imshow(X_fake[i])
    # save result image 
    filename = f'plot_{step+1}.png'
    pyplot.savefig(os.path.join(config.GAN_LOG_DIR,filename))
    pyplot.close()
    # save model checkpoint
    model_filename = f'model_{step+1}.h5'
    g_model.save(os.path.join(config.GAN_LOG_DIR,model_filename))
    logger.info(f">> Saved : {filename} , {model_filename} ")


def train(g_model, dataset, n_epoch=100, batch_size=16):
    n_patch = dc_model.output_shape[1]
    batch_per_epoch = (dataset[1].shape[0]*(dataset[1].shape[1]//2))//batch_size
    n_steps = n_epoch*batch_per_epoch
    plotlosses = PlotLosses(outputs=[MatplotlibPlot()], groups={'dss model' : ['dss_loss'], 'dsc model' : ['dsc_loss'], 'gan model' : ['total_loss', 'gen_loss', 'gans_loss', 'ganc_loss']})

    save_interval = 10
    log_interval = 1

    for i in range(n_steps):
        [X_cnt, X_stl, X_trn], ydc_real, yds_real = generate_real_samples(dataset, batch_size, n_patch)
        X_fake_trn, ydc_fake, yds_fake = generate_fake_samples(g_model, [X_cnt, X_stl], n_patch)
        # train style descriminator
        usXds_stl = np.concatenate((X_stl, X_stl))
        usXds_trn = np.concatenate((X_trn, X_fake_trn))
        usysd = np.concatenate((yds_real, yds_fake))
        Xds_stl, Xds_trn, yds = shuffle(usXds_stl, usXds_trn, usysd)
        ds_loss = ds_train_step(Xds_stl, Xds_trn, yds)
        #ds_loss2 = ds_train_step(X_stl, X_fake_trn, yds_fake)
        #train content descriminator
        usXdc_cnt = np.concatenate((X_cnt, X_cnt))
        usXdc_trn = np.concatenate((X_trn, X_fake_trn))
        usydc = np.concatenate((ydc_real, ydc_fake))
        Xdc_cnt, Xdc_trn, ydc = shuffle(usXdc_cnt, usXdc_trn, usydc)
        dc_loss = dc_train_step(Xdc_cnt, Xdc_trn, ydc)
        #dc_loss2 = dc_train_step(X_cnt, X_fake_trn, ydc_fake)
        #train GAN model
        gan_total_loss, gan_dss_loss, gan_dsc_loss, gen_loss = gan_train_step(X_cnt, X_stl, X_trn, ydc_real, yds_real)

        logger.info(f'[{i}/{n_steps}] : style descriminator total loss : {ds_loss} \n content descriminator total loss : {dc_loss} \n GAN total loss : {gan_total_loss} | GAN dss loss : {gan_dss_loss} | GAN dsc loss : {gan_dsc_loss}')
        # if (i+1) % (batch_per_epoch*log_interval) == 0:
        #     plotlosses.update({
        #         'dss_loss' : ds_loss,
        #         'dsc_loss' : dc_loss,
        #         'total_loss' : gan_total_loss,
        #         'gen_loss' : gen_loss,
        #         'gans_loss' : gan_dss_loss,
        #         'ganc_loss' : gan_dsc_loss
        #     })
        #     plotlosses.send()
        plotlosses.update({
                'dss_loss' : ds_loss,
                'dsc_loss' : dc_loss,
                'total_loss' : gan_total_loss,
                'gen_loss' : gen_loss,
                'gans_loss' : gan_dss_loss,
                'ganc_loss' : gan_dsc_loss
            })
        plotlosses.send()
        if (i+1) % (batch_per_epoch*save_interval) == 0:
            summarize_performance(i, g_model, dataset)
        if i == 200:
            summarize_performance(i, g_model, dataset)
            break


#%%

if __name__ == "__main__":
    #load dataset
    dataset = load_pixel_metrics(config.GAN_DATASET_DIR)
    #init models
    g_model = defing_generator(config.GAN_LATENT_SIZE, config.IMAGE_SHAPE)
    dc_model = define_cnt_descriminator()
    ds_base_model = define_style_descrminator(config.DESCS_LATENT_SIZE, config.IMAGE_SHAPE)
    ds_model = StyleNet(ds_base_model)
    gan_model = define_gan(g_model, dc_model, ds_model)

    #train model
    train(g_model, dataset, config.GAN_EPOCHS, config.GAN_BATCH_SIZE)


