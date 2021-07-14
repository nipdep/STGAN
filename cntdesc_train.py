

#%%
import config
from src.model.cntdesc_model import define_cnt_encoder, ContentNet
from src.support.loss_functions import pairWiseRankingLoss, MarginalAcc, triplet_loss

import os
import logging
import time
from datetime import datetime
import pathlib
import pandas as pd  
import numpy as np
import random
import tensorflow as tf  
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras import losses
from tensorflow.keras import metrics 
from tensorflow.keras.callbacks import TensorBoard
from matplotlib import pyplot as plt
from livelossplot import PlotLosses
from livelossplot.outputs import MatplotlibPlot

#%%
#tensorboard logger
logdir = config.LOG_DIR+ "/descc_pre_" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=logdir, histogram_freq=1, profile_batch=1)

# set logger
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

@tf.function
def train_step(st1_img1, st1_img2, st2_img1):
    with tf.GradientTape() as tape:
        vec1, vec2, vec3 = desc_model([st1_img1, st1_img2, st2_img1])
        #total_loss = triplet_loss(vec1, vec2, vec3, config.LOSS_THD)
        # vecc1 = tf.stack([vec1, vec2], axis=0)
        # vecc2 = tf.stack([vec1, vec3], axis=0)
        # lbls = tf.stack([tf.cast(tf.broadcast_to(1, shape=[vec1.shape[0]]), dtype=tf.bool), tf.cast(tf.broadcast_to(0, shape=[vec1.shape[0]]), dtype=tf.bool)], axis=1)
        loss1 = pairWiseRankingLoss(vec1, vec2, tf.cast(tf.broadcast_to(1, shape=[vec1.shape[0]]), dtype=tf.bool))
        loss2 = pairWiseRankingLoss(vec1, vec3, tf.cast(tf.broadcast_to(0, shape=[vec1.shape[0]]), dtype=tf.bool))
        total_loss = loss1+loss2
    grads = tape.gradient(total_loss, base_model.trainable_variables)

    opt.apply_gradients(zip(grads, base_model.trainable_variables))
    train_metrics.update_state(vec1, vec2, tf.cast(tf.broadcast_to(1, shape=[vec1.shape[0]]), dtype=tf.bool))
    train_metrics.update_state(vec1, vec3, tf.cast(tf.broadcast_to(0, shape=[vec1.shape[0]]), dtype=tf.bool))
    return total_loss

@tf.function
def val_step(st1_img1, st1_img2, st2_img1):
    with tf.GradientTape() as tape:
        vec1, vec2, vec3 = desc_model([st1_img1, st1_img2, st2_img1])
        #total_loss = triplet_loss(vec1, vec2, vec3, config.LOSS_THD)
        loss1 = pairWiseRankingLoss(vec1, vec2, tf.cast(tf.broadcast_to(1, shape=[vec1.shape[0]]), dtype=tf.bool))
        loss2 = pairWiseRankingLoss(vec1, vec3, tf.cast(tf.broadcast_to(0, shape=[vec1.shape[0]]), dtype=tf.bool))
        total_loss = loss1+loss2

    val_metrics.update_state(vec1, vec2, tf.cast(tf.broadcast_to(1, shape=[vec1.shape[0]]), dtype=tf.bool))
    val_metrics.update_state(vec1, vec3, tf.cast(tf.broadcast_to(0, shape=[vec1.shape[0]]), dtype=tf.bool))
    return total_loss


def train(epochs=3):
    tensorboard_callback.set_model(base_model)
    plotlosses = PlotLosses(outputs=[MatplotlibPlot()], groups={'loss' : ['tr_loss', 'val_loss'], 'accuracy' : ['tr_acc', 'val_acc']})
    for epoch in range(epochs):
        #print("\nStart of epoch %d" % (epoch,))
        start_time = time.time()
        
        # Iterate over the batches of the dataset.
        for step, (st1_img1, st1_img2, st2_img1) in enumerate(train_dt):
            train_loss = train_step(st1_img1, st1_img2, st2_img1)
        #print(f" Epoch [{epoch}] : relative accuracy : {train_acc[0]}, ranking loss : {train_acc[1]}")
        
        #Run a validation loop at the end of each epoch.
        for st1_img1, st1_img2, st2_img1 in val_dt:
            val_loss = val_step(st1_img1, st1_img2, st2_img1)

        tr_acc = train_metrics.result()
        val_acc = val_metrics.result()
        plotlosses.update({
            'tr_loss' : train_loss,
            'tr_acc' : tr_acc,
            'val_loss' : val_loss,
            'val_acc' : val_acc,
        })
        plotlosses.send()

        train_metrics.reset_states()
        val_metrics.reset_states()
    
        print("Time taken: %.2fs" % (time.time() - start_time))

#%%

if __name__ == '__main__':
    dt_mat = np.swapaxes(np.load(config.DESC_CNT_TRDT_DIR)['content'], 0, 1)
    dv_mat = np.swapaxes(np.load(config.DESC_CNT_VALDT_DIR)['content'][:500, ...], 0, 1)
    train_dataset = tf.data.Dataset.from_tensor_slices(dt_mat)
    val_dataset = tf.data.Dataset.from_tensor_slices(dv_mat)
    print(train_dataset.element_spec, val_dataset.element_spec)

    train_steps = 100
    lr_fn = tf.optimizers.schedules.PolynomialDecay(2e-3, train_steps, 1e-5, 2)
    opt = tf.optimizers.Adam(lr_fn)

    train_ds = train_dataset.map(lambda x: (x[0], x[1], x[2]),
                            num_parallel_calls=tf.data.AUTOTUNE,
                            deterministic=False)
    val_ds = val_dataset.map(lambda x: (x[0], x[1], x[2]),
                            num_parallel_calls=tf.data.AUTOTUNE,
                            deterministic=False)
    trainb_dt = train_ds.batch(batch_size=config.DESC_BATCH_SIZE)
    train_dt = trainb_dt.cache().prefetch(tf.data.AUTOTUNE)

    valb_dt = val_ds.batch(batch_size=config.DESC_BATCH_SIZE)
    val_dt = trainb_dt.cache().prefetch(tf.data.AUTOTUNE)

    base_model = define_cnt_encoder(config.DESCC_LATENT_SIZE, config.IMAGE_SHAPE)
    desc_model = ContentNet(base_model)

    #loss_func = tf.keras.losses.Hinge()
    train_metrics = MarginalAcc()
    val_metrics = MarginalAcc()
    #opt = tf.keras.optimizers.Adam(lr=config.DESCC_INIT_LR)
    train(config.DESCC_EPOCHS)

    filename = 'descc_wgt1.h5'
    base_model.save_weights(os.path.join(config.MODEL_DIR, filename))
    logger.info(f">> Saved : {filename}  ")

# %%
