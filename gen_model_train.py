##! usr/bin/python3
# %%

import config
from src.model.cntdesc_model import define_cnt_encoder, ContentNet
from src.model.stldesc_model import define_desc_encoder, StyleNet
from src.model.gen_model import define_generator
from src.support.loss_functions import pairWiseRankingLoss, MarginalAcc, triplet_loss
from src.model.stldesc_model import define_desc_encoder, StyleNet, define_stl_encoder

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
logdir = config.LOG_DIR+ "/gen_" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=logdir, histogram_freq=1, profile_batch=1)
run_opts = tf.compat.v1.RunOptions(report_tensor_allocations_upon_oom = True)

def SM_SSIMLoss(ref_img, gen_img):
    one = tf.cast(tf.broadcast_to(1, shape=ref_img.shape), dtype=tf.float32)
    two = tf.cast(tf.broadcast_to(2, shape=ref_img.shape), dtype=tf.float32)
    rescaled_ref_img = tf.abs(tf.divide(tf.add(one, ref_img), two))
    rescaled_gen_img = tf.abs(tf.divide(tf.add(one, gen_img), two))
    loss = tf.image.ssim_multiscale(ref_img, gen_img, max_val=2, filter_size=3)
    return tf.reduce_mean(loss)

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
    lower, higher, root_style_path, root_cnt_path, n = 1, 1100, './data/data/styleU', './data/data/MSO/MSOCntImg', 1000
    idx = np.random.choice(range(lower, higher), n, replace=False)
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
            stl_img = process_path(os.path.join(cnt_det))
            cnt_img = process_path(os.path.join(root_style_path, stl_det['path']))
            yield stl_img, cnt_img#, stl_det['style_code']
        except:
            print(f"Error in file {cnt_det} | {stl_det['path']}")
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

    ds = ds.map(lambda slt, cnt: (resize_and_rescale(slt), resize_and_rescale(cnt)),
                num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.cache()
    
    if shuffle:
        ds = ds.shuffle(1000)

    ds = ds.batch(16)
    return ds.prefetch(buffer_size=tf.data.AUTOTUNE)


dscLoss = tf.keras.losses.BinaryCrossentropy()
cntLoss = tf.keras.losses.MeanAbsoluteError()

def add_cnt_loss(dis_loss, gen_loss):
    return dis_loss + config.LAMBDAC*gen_loss

def add_style_loss(dis_loss, gen_loss):
    return dis_loss + config.LAMBDAS*gen_loss

def genLoss(dss_loss, dsc_loss, gen_loss):
    gan_alpha = config.GAN_ALPHA
    gan_beta = config.GAN_BETA
    one = 1

    tot_loss = gan_alpha*(gan_beta*dss_loss+(one-gan_beta)*dsc_loss)+(one-gan_alpha)*gen_loss
    return tot_loss

@tf.function
def train_step(cnt_in, style_in):
    with tf.GradientTape() as gen_tape:
        gen_img = gen_model([cnt_in, style_in])
        cnt_vec, gen_vec = cnt_base_model(cnt_in), cnt_base_model(gen_img)
        stl_vec, gen_vec1 = stl_base_model(style_in), stl_base_model(gen_img)
        
        cnt_loss = pairWiseRankingLoss(cnt_vec, gen_vec, tf.cast(tf.broadcast_to(1, shape=[cnt_vec.shape[0]]), dtype=tf.bool))
        stl_loss = stlLoss(stl_vec, gen_vec1)
        gen_loss = cntLoss(cnt_in, gen_img)
        total_loss = genLoss(stl_loss,cnt_loss, gen_loss)

    grads = gen_tape.gradient(total_loss, gen_model.trainable_variables)
    opt.apply_gradients(zip(grads, gen_model.trainable_variables))
    stl_metrics.update_state(stl_vec, gen_vec1)
    cnt_metrics.update_state(cnt_vec, gen_vec, tf.cast(tf.broadcast_to(1, shape=[cnt_vec.shape[0]]), dtype=tf.bool))
    return total_loss, gen_loss, cnt_loss, stl_loss

def load_pixel_metrics(filename):
    full_mat = np.load(filename)
    style_pixels = (full_mat['style']-127.5)/127.5
    content_pixels = (full_mat['cotent']-127.5)/127.5
    transfer_mat  = (full_mat['transfers']-127.5)/127.5
    return style_pixels, content_pixels, transfer_mat

def generate_samples(dataset, n_samples, patch_shape):
    style, content= dataset.take(n_samples)
    return [cnt_pixels, style_pixels, mat_pixels], y_dc, y_ds

def generate_fake_samples(g_model, samples, patch_shape):
    cnt_img, style_img = samples
    X = g_model([cnt_img, style_img])
    y_dc = zeros((len(X), patch_shape, patch_shape, 1))
    y_ds = zeros((len(X)))
    return X, y_dc, y_ds

def summarize_performance(step, g_model, dataset, n_samples=3):
    stl_sample, cnt_sample = dataset
    gen_sample = g_model([stl_sample, cnt_sample])
    #rescale pixels values
    X_cnt = (cnt_sample+1)/2
    X_stl = (stl_sample+1)/2
    X_trn = (gen_sample+1)/2
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
    filename = f'plot_{step+1}.png'
    pyplot.savefig(os.path.join(config.GAN_LOG_DIR,filename))
    pyplot.close()
    # save model checkpoint
    # if step % 100:
    #     model_filename = f'model_{step+1}.h5'
    #     g_model.save(os.path.join(config.GAN_LOG_DIR,model_filename))
    #     logger.info(f">> Saved : {model_filename} ")


def train(epochs=3):
    #tensorboard_callback.set_model(desc_pre_model)
    plotlosses = PlotLosses(outputs=[MatplotlibPlot()], groups={'loss' : ['total_loss', 'gen_loss', 'stl_loss', 'cnt_loss'], 'accuracy' : ['stl_acc', 'cnt_acc']})
    for epoch in range(epochs):
        start_time = time.time()
        
        # Iterate over the batches of the dataset.
        for step, (cnt_batch, style_batch) in enumerate(train_dataset):
            total_loss, gen_loss, cnt_loss, stl_loss = train_step(cnt_batch, style_batch)

        # Run a validation loop at the end of each epoch.
        # for x_batch_val, y_batch_val in val_dataset:
        #     val_loss = val_step(x_batch_val, y_batch_val)

        stl_acc = stl_metrics.result()
        cnt_acc = cnt_metrics.result()
        plotlosses.update({
            'total_loss' : total_loss,
            'gen_loss' : gen_loss,
            'stl_loss' : stl_loss,
            'cnt_loss' : cnt_loss,
            'stl_acc' : stl_acc,
            'cnt_acc' : cnt_acc
        })
        plotlosses.send()

        stl_metrics.reset_states()
        cnt_metrics.reset_states()
        # val_acc = val_metrics.result()
        # val_metrics.reset_states()
        # print("Validation acc: %.4f" % (float(val_acc),))
        print("Time taken: %.2fs" % (time.time() - start_time))
        summarize_performance(epoch, gen_model, [cnt_batch, style_batch], 5)


# def train(g_model, dataset, n_epoch=100, batch_size=16):
#     n_patch = dc_model.output_shape[1]
#     batch_per_epoch = (dataset[1].shape[0]*(dataset[1].shape[1]//2))//batch_size
#     n_steps = n_epoch*batch_per_epoch
#     plotlosses = PlotLosses(outputs=[MatplotlibPlot()], groups={'dss model' : ['dss_loss'], 'dsc model' : ['dsc_loss'], 'gan model' : ['gen_loss']})

#     save_interval = 10
#     log_interval = 1

#     for i in range(n_steps):
#         [X_cnt, X_stl, X_trn], ydc_real, yds_real = generate_real_samples(dataset, batch_size, n_patch)
#         X_fake_trn, ydc_fake, yds_fake = generate_fake_samples(g_model, [X_cnt, X_stl], n_patch)
#         # train style descriminator
#         usXds_stl = np.concatenate((X_stl, X_stl))
#         usXds_trn = np.concatenate((X_trn, X_fake_trn))
#         usysd = np.concatenate((yds_real, yds_fake))
#         Xds_stl, Xds_trn, yds = shuffle(usXds_stl, usXds_trn, usysd)
#         #train content descriminator
#         usXdc_cnt = np.concatenate((X_cnt, X_cnt))
#         usXdc_trn = np.concatenate((X_trn, X_fake_trn))
#         usydc = np.concatenate((ydc_real, ydc_fake))
#         Xdc_cnt, Xdc_trn, ydc = shuffle(usXdc_cnt, usXdc_trn, usydc)

#         #train GAN model
#         gen_loss, dc_loss, ds_loss = train_step(X_cnt, X_stl, X_trn, ydc_fake, yds_fake, Xds_stl, Xds_trn, yds, Xdc_cnt, Xdc_trn, ydc)
    

#         #logger.info(f'[{i}/{n_steps}] : style descriminator total loss : {ds_loss} \n content descriminator total loss : {dc_loss} \n GAN total loss : {gan_total_loss} | GAN dss loss : {gan_dss_loss} | GAN dsc loss : {gan_dsc_loss}')
#         if i % 10 == 0: 
#             plotlosses.update({
#                     'dss_loss' : ds_loss,
#                     'dsc_loss' : dc_loss,
#                     'gen_loss' : gen_loss,
#                 })
#             plotlosses.send()
#         if (i+1) % (batch_per_epoch*save_interval) == 0:
#             summarize_performance(i, g_model, dataset)
#         if i % 100 == 0:
#             summarize_performance(i, g_model, dataset)
#             if i == config.GAN_BP:
#                 break

#%%
if __name__ == "__main__":
    #load dataset
    stenc_df = pd.read_csv('./data/data/styleU/StyleEnc.csv', index_col=0)
    train_path = pathlib.Path(os.path.join(config.DESC_ROOT_DIR,'train'))
    train_ds = tf.data.Dataset.from_generator(
        train_gen,
        output_signature=(
            tf.TensorSpec(shape=(128,128, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(128,128,3), dtype=tf.float32)
            #tf.TensorSpec(shape=(), dtype=tf.int32)
        )

    )
    train_dataset = prepare(train_ds, shuffle=True)

    cnt_model_dir = "./data/models/descc_wgt1.h5"
    stl_model_dir = "./data/models/dess_m6.h5"
    #stl_base_model = define_stl_encoder(config.DESCS_LATENT_SIZE, config.IMAGE_SHAPE)
    stl_base_model = load_model(stl_model_dir)
    cnt_base_model = define_cnt_encoder(config.DESCC_LATENT_SIZE, config.IMAGE_SHAPE)
    cnt_base_model.load_weights(cnt_model_dir)

    gen_model = define_generator(cnt_base_model, stl_base_model, (128, 128, 3))

    train_steps = 100
    lr_fn = tf.optimizers.schedules.PolynomialDecay(1e-3, train_steps, 1e-5, 2)
    opt = tf.optimizers.Adam(lr_fn)

    #stl_metrics = MarginalAcc()
    stlLoss = tf.keras.losses.MeanSquaredError()
    stl_metrics = tf.keras.metrics.MeanAbsoluteError()
    cnt_metrics = MarginalAcc()
    #train model
    train(config.GAN_EPOCHS)



# %%
