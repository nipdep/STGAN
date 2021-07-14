##! usr/bin/python3
# %%

import config
from src.model.desc_model import define_style_descrminator, StyleNet
from src.model.gan_model import define_cnt_descriminator, define_gan, define_generator
#from src.model.wavelet_gan_model import define_cnt_descriminator, define_gan, define_generator

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
from tensorflow.keras.callbacks import TensorBoard
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
run_opts = tf.compat.v1.RunOptions(report_tensor_allocations_upon_oom = True)

def pairWiseRankingLoss(y_ref, y_style, label):
    m  = tf.cast(tf.broadcast_to(config.LOSS_THD, shape=[y_ref.shape[0], ]), dtype=tf.float32)
    u  = tf.cast(tf.broadcast_to(0, shape=[y_ref.shape[0], ]), dtype=tf.float32)
    i  = tf.cast(tf.broadcast_to(1, shape=[y_ref.shape[0], ]), dtype=tf.float32)
    y = tf.cast(label, dtype=tf.float32)
    dist = tf.math.abs(tf.keras.losses.cosine_similarity(y_ref,y_style))
    loss = tf.math.multiply(y,dist) + tf.math.multiply((i-y),tf.reduce_max(tf.stack([u,m-dist]), axis=0))
    return tf.cast(tf.reduce_mean(loss), dtype=tf.float32)

def mixLoss(ref_img, gen_img):
    one = tf.cast(tf.broadcast_to(1, shape=ref_img.shape), dtype=tf.float32)
    two = tf.cast(tf.broadcast_to(2, shape=ref_img.shape), dtype=tf.float32)
    rescaled_ref_img = tf.abs(tf.divide(tf.add(one, ref_img), two))
    rescaled_gen_img = tf.abs(tf.divide(tf.add(one, gen_img), two))
    l1_loss = tf.norm(ref_img-gen_img, ord=1, axis=0)/ref_img.shape[0]
    ms_ssim_loss = tf.reduce_mean(tf.image.ssim_multiscale(rescaled_ref_img, rescaled_gen_img, max_val=1, filter_size=3))
    alpha = tf.cast(config.GEN_LOSS_ALPHA, dtype=tf.float32)
    total_loss = alpha*ms_ssim_loss + (1-alpha)*l1_loss
    return tf.cast(total_loss, dtype=tf.float32)

def ganLoss(dss_loss, dsc_loss, gen_loss):
    gan_alpha = config.GAN_ALPHA
    gan_beta = config.GAN_BETA
    one = 1

    tot_loss = gan_alpha*(gan_beta*dss_loss+(one-gan_beta)*dsc_loss)+(one-gan_alpha)*gen_loss
    return tot_loss

dscLoss = tf.keras.losses.BinaryCrossentropy()
cntLoss = tf.keras.losses.MeanAbsoluteError()

gen_opt = tf.keras.optimizers.Adam(lr=0.002)
ds_opt = tf.keras.optimizers.Adam(lr=0.02)
dc_opt = tf.keras.optimizers.Adam(lr=0.02)

def add_cnt_loss(dis_loss, gen_loss):
    return dis_loss + config.LAMBDAC*gen_loss

def add_style_loss(dis_loss, gen_loss):
    return dis_loss + config.LAMBDAS*gen_loss

@tf.function
def train_step(cnt_in, style_in, trans_in, cnt_fake, style_fake, Xds_stl, Xds_trn, yds, Xdc_cnt, Xdc_trn, ydc):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as discs_tape, tf.GradientTape() as discc_tape:
        gen_out, dss_out, dst_out, cnt_out = gan_model([cnt_in, style_in])
        dss_loss = pairWiseRankingLoss(dss_out, dst_out, style_fake)
        dsc_loss = dscLoss(cnt_fake, cnt_out)
        gen_loss = tf.cast(tf.math.abs(cntLoss(trans_in, gen_out)), dtype=tf.float32)

        ref_out, trans_out = ds_model([Xds_stl, Xds_trn])
        ds_loss = pairWiseRankingLoss(ref_out, trans_out, yds)

        logits = dc_model([Xdc_cnt, Xdc_trn])
        dc_loss = dscLoss(ydc, logits)

        total_style_loss = add_style_loss(ds_loss, dss_loss)
        total_cnt_loss = add_cnt_loss(dc_loss, dsc_loss)
        total_gen_loss = ganLoss(dss_loss, dsc_loss, gen_loss)
        #total_gen_loss = mixLoss(trans_in, gen_out)

    generator_grads = gen_tape.gradient(total_gen_loss, g_model.trainable_variables)
    cnt_disc_grads = discc_tape.gradient(total_cnt_loss, dc_model.trainable_variables)
    style_disc_grads = discs_tape.gradient(total_style_loss, ds_base_model.trainable_variables)

    gen_opt.apply_gradients(zip(generator_grads, g_model.trainable_variables))
    dc_opt.apply_gradients(zip(cnt_disc_grads, dc_model.trainable_variables))
    ds_opt.apply_gradients(zip(style_disc_grads,  ds_base_model.trainable_variables))
    return gen_loss, total_cnt_loss, total_style_loss

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
    plotlosses = PlotLosses(outputs=[MatplotlibPlot()], groups={'dss model' : ['dss_loss'], 'dsc model' : ['dsc_loss'], 'gan model' : ['gen_loss']})

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
        #train content descriminator
        usXdc_cnt = np.concatenate((X_cnt, X_cnt))
        usXdc_trn = np.concatenate((X_trn, X_fake_trn))
        usydc = np.concatenate((ydc_real, ydc_fake))
        Xdc_cnt, Xdc_trn, ydc = shuffle(usXdc_cnt, usXdc_trn, usydc)

        #train GAN model
        gen_loss, dc_loss, ds_loss = train_step(X_cnt, X_stl, X_trn, ydc_fake, yds_fake, Xds_stl, Xds_trn, yds, Xdc_cnt, Xdc_trn, ydc)
    

        #logger.info(f'[{i}/{n_steps}] : style descriminator total loss : {ds_loss} \n content descriminator total loss : {dc_loss} \n GAN total loss : {gan_total_loss} | GAN dss loss : {gan_dss_loss} | GAN dsc loss : {gan_dsc_loss}')
        if i % 10 == 0: 
            plotlosses.update({
                    'dss_loss' : ds_loss,
                    'dsc_loss' : dc_loss,
                    'gen_loss' : gen_loss,
                })
            plotlosses.send()
        if (i+1) % (batch_per_epoch*save_interval) == 0:
            summarize_performance(i, g_model, dataset)
        if i % 100 == 0:
            summarize_performance(i, g_model, dataset)
            if i == config.GAN_BP:
                break


if __name__ == "__main__":
    #load dataset
    dataset = load_pixel_metrics(config.GAN_DATASET_DIR)
    #init models
    g_model = define_generator(config.GAN_LATENT_SIZE, config.IMAGE_SHAPE)
    dc_model = define_cnt_descriminator()
    ds_base_model = define_style_descrminator(config.DESCS_LATENT_SIZE, config.IMAGE_SHAPE)
    ds_model = StyleNet(ds_base_model)
    gan_model = define_gan(g_model, dc_model, ds_model)

    #train model
    train(g_model, dataset, config.GAN_EPOCHS, config.GAN_BATCH_SIZE)



# %%
