##! usr/bin/python3
# %%

import config
from src.model.cntdesc_model import define_cnt_encoder, ContentNet
from src.model.gen_model import define_generator, define_gan
from src.support.loss_functions import pairWiseRankingLoss, MarginalAcc, triplet_loss
from src.model.stldesc_model import define_desc_encoder, StyleNet, define_stl_encoder, define_stl_regressor, stl_encoder

#from src.model.wavelet_gan_model import define_cnt_descriminator, define_gan, define_generator

import os 
import logging
import time
import random
from datetime import datetime
from livelossplot import outputs
import tensorflow as tf
import numpy as np
import pandas as pd
from numpy import load, zeros, ones
import pathlib
from numpy.random import randint
from sklearn.utils import shuffle
import tensorflow_addons as tfa
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.callbacks import TensorBoard
import tensorflow.keras.layers.experimental.preprocessing as prep
from tensorflow.keras.models import load_model
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
# logdir = config.LOG_DIR+ "/rgan_" + datetime.now().strftime("%Y%m%d-%H%M%S")
# tensorboard_callback = TensorBoard(log_dir=logdir, histogram_freq=1, profile_batch=1)
run_opts = tf.compat.v1.RunOptions(report_tensor_allocations_upon_oom = True)

# @tf.function
# def train_step(cnt_in, style_in, stly):
#     with tf.GradientTape() as gen_tape:
#         gen_img = gen_model([cnt_in, style_in])
#         cnt_vec, gen_vec = cnt_base_model(cnt_in), cnt_base_model(gen_img)
#         gen_vec1 = stl_base_model(gen_img)
        
#         cnt_loss = pairWiseRankingLoss(cnt_vec, gen_vec, tf.cast(tf.broadcast_to(1, shape=[cnt_vec.shape[0]]), dtype=tf.bool))
#         stl_loss = stlLoss(stly, gen_vec1)
#         gen_loss = genLoss(cnt_in, gen_img)
#         total_loss = totalLoss(stl_loss,cnt_loss, gen_loss)

#     grads = gen_tape.gradient(total_loss, gen_model.trainable_variables)
#     opt.apply_gradients(zip(grads, gen_model.trainable_variables))
#     #stl_metrics.update_state(gen_vec1, stly)
#     #cnt_metrics.update_state(cnt_vec, gen_vec, tf.cast(tf.broadcast_to(1, shape=[cnt_vec.shape[0]]), dtype=tf.bool))
#     return total_loss, gen_loss, cnt_loss, stl_loss

def process_img(img):
  img = tf.image.decode_jpeg(img, channels=3) 
  img = tf.image.convert_image_dtype(img, tf.float32) 
  return tf.image.resize(img, config.IMAGE_SIZE)

def process_path(file_path):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    #print(fp)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, size=(128, 128))
    return img

def train_gen():
    lower, higher, root_style_path, root_cnt_path, n = 1, 1100, './data/data/styleU', './data/data/MSO/MSOCntImg', 2016
    idx = np.random.choice(range(lower, higher), n, replace=True)
    for i in idx:
        #i = random.randint(lower, higher)
        random_num = random.randint(1, stenc_df.shape[0])
        # random_bool = random.randint(0,1)
        # if random_bool:
        #     if random_num == int(i):
        #         random_num = random.randint(lower, higher)
        # else:
        #     random_num = max(random.randint(1,10), int(i)-5)
        cnt_det = os.path.join(root_cnt_path, f'{i}.jpg')
        stl_det = stenc_df.loc[random_num, ['path', 'style_code']]

        # label = 0
        # if img1_det['style_code'] == img2_det['style_code']:
        #     label = 1
        #print(os.path.join(root_path, img1_det['path']), os.path.join(root_path, img2_det['path']))
        try :
            cnt_img = process_path(cnt_det)
            stl_path = os.path.join(root_style_path, stl_det['path'])
            stl_img = process_path(stl_path)
            yield stl_img, cnt_img, stl_det['style_code']
        except:
            # print(e)
            print(f"Error in file {cnt_det} | {stl_path}")
            continue

# image resize and rescale pipeline
resize_and_rescale = tf.keras.Sequential([
    prep.Resizing(config.IMG_HEIGHT, config.IMG_WIDTH),
    prep.Normalization()
])

def prepare(ds, shuffle=False):
    # ds = ds.map(lambda x: tf.py_function(process_path, [x], [tf.float32, tf.float32, tf.int32]),
    #                         num_parallel_calls=tf.data.AUTOTUNE)

    # ds = ds.map(lambda x1, x2, y: (process_path(x1), process_path(x2), y),
    #             num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.map(lambda slt, cnt, y: (resize_and_rescale(slt), resize_and_rescale(cnt), y),
                num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.cache()
    
    if shuffle:
        ds = ds.shuffle(1000)

    ds = ds.batch(16, drop_remainder=True)
    return ds.prefetch(buffer_size=tf.data.AUTOTUNE)

def summarize_performance(step, g_model, dataset, n_samples=3):
    cnt_sample, stl_sample = dataset
    gen_sample = g_model([cnt_sample, stl_sample])
    #rescale pixels values
    # X_cnt = (cnt_sample+1)/2
    # X_stl = (stl_sample+1)/2
    # X_trn = (gen_sample+1)/2
    X_cnt = cnt_sample
    X_stl = stl_sample
    X_trn = gen_sample
    # plot samples
    for i in range(n_samples):
        pyplot.subplot(3, n_samples, 1 + i)
        pyplot.axis('off')
        pyplot.imshow(X_cnt[i])
    for i in range(n_samples):
        pyplot.subplot(3, n_samples, 1 + n_samples + i)
        pyplot.axis('off')
        pyplot.imshow(X_stl[i])
    for i in range(n_samples):
        pyplot.subplot(3, n_samples, 1 + 2*n_samples + i)
        pyplot.axis('off')
        pyplot.imshow(X_trn[i])
    # save result image 
    filename = f'plot_rg{step+1}.png'
    pyplot.savefig(os.path.join(config.GAN_LOG_DIR,filename))
    pyplot.close()
    # save model checkpoint
    # model_filename = f'model_{step+1}.h5'
    # g_model.save(os.path.join(config.GAN_LOG_DIR,model_filename))
    # logger.info(f">> Saved : {filename} , {model_filename} ")


def ganLoss(dss_loss, dsc_loss, gen_loss):
    gan_alpha = config.GAN_ALPHA
    gan_beta = config.GAN_BETA
    one = 1

    tot_loss = dss_loss+dsc_loss+gen_loss
    return tot_loss

def add_cnt_loss(dis_loss, gen_loss):
    return dis_loss + config.LAMBDAC*gen_loss

def add_style_loss(dis_loss, gen_loss):
    return dis_loss + config.LAMBDAS*gen_loss

@tf.function
def train_step(cnt_in, style_in, y_in, Xds_stl, Xds_trn, yds, Xdc_cnt, Xdc_trn, ydc):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as discs_tape, tf.GradientTape() as discc_tape:
        gen_out, dss_out, dst_out, cnt_out, dct_out = gan_model([cnt_in, style_in])
        dsc_loss = pairWiseRankingLoss(cnt_out, dct_out, tf.cast(tf.broadcast_to(1, shape=[cnt_out.shape[0]]), dtype=tf.bool),1)
        gen_loss = tf.cast(cntLoss(cnt_in, gen_out), dtype=tf.float32)

        similarity1 = tf.einsum(
            "ae,pe->ap", dss_out, dst_out
        )
        temp = 0.2
        similarity1 /= temp
        dss_loss = stlLoss(y_in, similarity1)

        ref_stl_out, trans_stl_out = ds_model([Xds_stl, Xds_trn])
        similarity2 = tf.einsum(
            "ae,pe->ap", ref_stl_out, trans_stl_out
        )
        similarity2 /= temp
        ds_loss = stlLoss(yds, similarity2)

        ref_cnt_out, trans_stl_out = dc_model([Xdc_cnt, Xdc_trn])
        dc_loss = pairWiseRankingLoss(ref_cnt_out, trans_stl_out, ydc, 1)

        # total_style_loss = add_style_loss(ds_loss, dss_loss)
        # total_cnt_loss = add_cnt_loss(dc_loss, dsc_loss)
        total_gen_loss = ganLoss(dss_loss, dsc_loss, gen_loss)
        #total_gen_loss = mixLoss(trans_in, gen_out)

    generator_grads = gen_tape.gradient(total_gen_loss, gen_model.trainable_variables)
    cnt_disc_grads = discc_tape.gradient(dc_loss, cnt_base_model.trainable_variables)
    style_disc_grads = discs_tape.gradient(ds_loss, stl_base_model.trainable_variables)

    gen_opt.apply_gradients(zip(generator_grads, gen_model.trainable_variables))
    dc_opt.apply_gradients(zip(cnt_disc_grads, cnt_base_model.trainable_variables))
    ds_opt.apply_gradients(zip(style_disc_grads,  stl_base_model.trainable_variables))
    return total_gen_loss, dc_loss, ds_loss

def generate_real_samples(dataset):
    style, content, y = dataset
    # cnt_idxs = random.sample(range(style.shape[1]), n_samples)
    # style_idxs = np.random.randint(0, style.shape[0], n_samples)

    # cnt_pixels = content[cnt_idxs]
    # style_pixels = style[style_idxs]
    # mat_pixels = trans[style_idxs, cnt_idxs, ...]

    # y_dc = ones((n_samples, patch_shape, patch_shape, 1))
    y_dc = ones((y.shape[0]))
    return [style, content], y_dc, y

def generate_fake_samples(g_model, samples):
    cnt_img, style_img = samples
    X = g_model([cnt_img, style_img])
    # y_dc = zeros((len(X), patch_shape, patch_shape, 1))
    y_dc = zeros((len(X)))
    return X, y_dc

def train(g_model, dataset, n_epoch=100, batch_size=16):
    batch_per_epoch = 2016//batch_size
    n_steps = n_epoch*batch_per_epoch
    plotlosses = PlotLosses(outputs=[MatplotlibPlot()], groups={'dss model' : ['dss_loss'], 'dsc model' : ['dsc_loss'], 'gan model' : ['gen_loss']})

    # save_interval = 10
    # log_interval = 1

    for epoch in range(n_epoch):
        start_time = time.time()
        
        # Iterate over the batches of the dataset.
        for step, (style_batch, cnt_batch, stly_batch) in enumerate(train_dataset):
            # total_loss, gen_loss, cnt_loss, stl_loss = train_step(cnt_batch, style_batch, stly_batch)
            if step % 2 == 0:
                [X_stl, X_cnt], ydc_real, yds_real = generate_real_samples((style_batch, cnt_batch, stly_batch))
                X_fake_trn, ydc_fake= generate_fake_samples(g_model, [X_cnt, X_stl])
                # train style descriminator
                # usXds_stl = np.concatenate((X_stl, X_stl))
                # usXds_trn = np.concatenate((X_trn, X_fake_trn))
                # usysd = np.concatenate((yds_real, yds_fake))
                # Xds_stl, Xds_trn, yds = shuffle(usXds_stl, usXds_trn, usysd)
                Xds_stl, Xds_trn, yds = X_stl, X_fake_trn, yds_real
                #train content descriminator
                usXdc_cnt = np.concatenate((X_cnt, X_stl))
                usXdc_trn = np.concatenate((X_stl, X_fake_trn))
                usydc = np.concatenate((ydc_real, ydc_fake))
                Xdc_cnt, Xdc_trn, ydc = shuffle(usXdc_cnt, usXdc_trn, usydc)
                continue
            else:
                #train GAN model
                gan_loss, dc_loss, ds_loss = train_step(style_batch, cnt_batch, stly_batch, Xds_stl, Xds_trn, yds, Xdc_cnt, Xdc_trn, ydc)
        

        #logger.info(f'[{i}/{n_steps}] : style descriminator total loss : {ds_loss} \n content descriminator total loss : {dc_loss} \n GAN total loss : {gan_total_loss} | GAN dss loss : {gan_dss_loss} | GAN dsc loss : {gan_dsc_loss}')
        # print(f'[{i}/{n_steps}] : style descriminator total loss : {ds_loss} \n content descriminator total loss : {dc_loss} \n GAN total loss : {gan_loss}')
         
        plotlosses.update({
                'dss_loss' : ds_loss,
                'dsc_loss' : dc_loss,
                'gen_loss' : gan_loss,
            })
        plotlosses.send()
        # if (i+1) % (batch_per_epoch*save_interval) == 0:
        #     summarize_performance(i, g_model, dataset)
        summarize_performance(epoch, g_model, (cnt_batch, style_batch))
        print("Time taken: %.2fs" % (time.time() - start_time))
            # if i == config.GAN_BP:
            #     break

#%%

if __name__ == "__main__":
    #load dataset
    stenc_df = pd.read_csv('./data/data/styleU/StyleEnc.csv', index_col=0)
    #dataset = load_pixel_metrics(config.GAN_DATASET_DIR)
    train_ds = tf.data.Dataset.from_generator(
        train_gen,
        output_signature=(
            tf.TensorSpec(shape=(128,128, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(128,128, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32)
        )

    )
    train_dataset = prepare(train_ds, shuffle=True)

    stlLoss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    dscLoss = tf.keras.losses.BinaryCrossentropy()
    cntLoss = tf.keras.losses.MeanAbsoluteError()

    gen_opt = tf.keras.optimizers.Adam(1e-4)
    ds_opt = tf.keras.optimizers.Adam(1e-5)
    dc_opt = tf.keras.optimizers.Adam(1e-5)

    #init models
    stl_base_model = stl_encoder(config.DESCS_LATENT_SIZE, config.IMAGE_SHAPE)
    cnt_base_model = define_cnt_encoder(config.DESCC_LATENT_SIZE, config.IMAGE_SHAPE)

    gen_model = define_generator(cnt_base_model, stl_base_model)
    dc_model = ContentNet(cnt_base_model)
    ds_model = StyleNet(stl_base_model)
    gan_model = define_gan(gen_model, dc_model, ds_model)

    #train model
    train(gen_model, train_dataset, 100, config.GAN_BATCH_SIZE)



# %%
