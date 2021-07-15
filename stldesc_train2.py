#! usr/bin/python3
#%%
import config
from src.model.stldesc_model import define_stl_encoder, EmbStyleNet
from src.support.loss_functions import pairWiseRankingLoss, MarginalAcc, triplet_loss


import os
import logging
import time
import math
from tqdm import tqdm
from datetime import datetime
import pathlib
import pandas as pd 
import numpy as np 
import random
import tensorflow.keras.backend as K
import tensorflow as tf  
import tensorflow.keras.layers.experimental.preprocessing as prep
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras import losses
from tensorflow.keras import metrics 
from tensorflow.keras.callbacks import TensorBoard
from matplotlib import pyplot as plt
from livelossplot import PlotLosses
from livelossplot.outputs import MatplotlibPlot

#tf.executing_eagerly()
#%%
#tensorboard logger
logdir = config.LOG_DIR+ "/desc_pre_" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=logdir, histogram_freq=1, profile_batch=1)

# tf.profiler.experimental.server.start(6009)

# set logger
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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
    lower, higher, root_path, n = 1, 2923, './data/data/StyleDataset', 2900
    idx = np.array(range(lower, min(higher, lower+n)))
    for i in idx:
        #i = random.randint(lower, higher)
        random_num = random.randint(lower, higher)
        random_bool = random.randint(0,1)
        if random_bool:
            if random_num == int(i):
                random_num = random.randint(lower, higher)
        else:
            random_num = max(random.randint(1,10), int(i)-5)
        img1_det = stenc_df.loc[i, ['path', 'style_code']]
        img2_det = stenc_df.loc[random_num, ['path', 'style_code']]

        # label = 0
        # if img1_det['style_code'] == img2_det['style_code']:
        #     label = 1
        #print(os.path.join(root_path, img1_det['path']), os.path.join(root_path, img2_det['path']))
        try :
            img1 = process_path(os.path.join(root_path, img1_det['path']))
            img2 = process_path(os.path.join(root_path, img2_det['path']))
            yield img1, img2, img1_det['style_code']-1, img2_det['style_code']-1
        except:
            print(f"Error in file {img1_det['path']}")
            continue

def val_gen():
    lower, higher, root_path, n = 2923, 3164, './data/data/StyleDataset', 200
    # idx = np.random.choice(range(lower, higher), n, replace=False, seed=111)
    # for i in idx:
    idx = np.array(range(lower, min(higher, lower+n)))
    for i in idx:
        #i = random.randint(lower, higher)
        random_num = random.randint(lower, higher)
        random_bool = random.randint(0,1)
        if random_bool:
            if random_num == int(i):
                random_num = random.randint(lower, higher)
        else:
            random_num = max(random.randint(1,10), int(i)-5)
        img1_det = stenc_df.loc[i, ['path', 'style_code']]
        img2_det = stenc_df.loc[random_num, ['path', 'style_code']]

        # label = 0
        # if img1_det['style_code'] == img2_det['style_code']:
        #     label = 1
        #print(os.path.join(root_path, img1_det['path']), os.path.join(root_path, img2_det['path']))
        try :
            img1 = process_path(os.path.join(root_path, img1_det['path']))
            img2 = process_path(os.path.join(root_path, img2_det['path']))
            yield img1, img2, img1_det['style_code']-1, img2_det['style_code']-1
        except:
            print(f"Error in file {img1_det['path']}")
            continue

# image resize and rescale pipeline
resize_and_rescale = tf.keras.Sequential([
    prep.Resizing(config.IMG_HEIGHT, config.IMG_WIDTH),
    prep.Rescaling(scale=1./127.5, offset=-1)
])

# image augmentation pipeline
data_augmentation = tf.keras.Sequential([
    prep.RandomContrast(0.2),
    prep.RandomFlip("horizontal_and_vertical"),
    prep.RandomCrop(config.IMG_HEIGHT, config.IMG_WIDTH),
    prep.RandomRotation(0.3, fill_mode='nearest', interpolation='bilinear'),
    prep.RandomZoom(height_factor=(-0.2, 0.2), width_factor=(-0.2, 0.2), fill_mode='nearest', interpolation='bilinear')
])

# data_augmentation = tf.keras.Sequential([
#   prep.RandomFlip("horizontal_and_vertical"),
#   prep.RandomRotation(0.2),
# ])

def prepare(ds, shuffle=False, augment=False):
    # ds = ds.map(lambda x: tf.py_function(process_path, [x], [tf.float32, tf.float32, tf.int32]),
    #                         num_parallel_calls=tf.data.AUTOTUNE)

    # ds = ds.map(lambda x1, x2, y: (process_path(x1), process_path(x2), y),
    #             num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.map(lambda x1, x2, y1, y2: (resize_and_rescale(x1), resize_and_rescale(x2), y1, y2),
                num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.cache()
    
    if shuffle:
        ds = ds.shuffle(1000)

    ds = ds.batch(16)

    if augment:
        ds = ds.map(lambda x1, x2, y1, y2: (data_augmentation(x1, training=True), data_augmentation(x2, training=True), y1, y2), 
                    num_parallel_calls=tf.data.AUTOTUNE)
    
    return ds.prefetch(buffer_size=tf.data.AUTOTUNE)

def get_loss(vec1t, vec1f, vec2t, vec2f):
    
    norm1, norm2, norm3, norm4 = tf.norm(vec1t, axis=0, ord=1) , tf.norm(vec1f, axis=0, ord=1) , tf.norm(vec2t, axis=0, ord=1) , tf.norm(vec2f, axis=0, ord=1)
    u  = tf.cast(tf.broadcast_to(0, shape=norm2.shape), dtype=tf.float32)
    norm2, norm4 = tf.math.reduce_max(tf.stack([u,0.2-norm2]), axis=0), tf.math.reduce_max(tf.stack([u,0.2-norm4]), axis=0)
    return tf.math.reduce_mean(norm1+norm2+norm3+norm4) 


@tf.function
def train_step(ref_in, style_in, ref_lbl, stl_lbl):
    with tf.GradientTape() as tape:
        #ref_out, style_out = desc_pre_model([ref_in, style_in])
        vec1t, vec1f, vec2t, vec2f = desc_pre_model([ref_in, style_in, ref_lbl, stl_lbl])
        loss = get_loss(vec1t, vec1f, vec2t, vec2f)
    grads = tape.gradient(loss, base_model.trainable_variables)
    opt.apply_gradients(zip(grads, base_model.trainable_variables))
    #train_metrics.update_state(ref_out, style_out, label_in)
    return loss

@tf.function
def val_step(ref_in, style_in, label_in):
    with tf.GradientTape() as tape:
        ref_out, style_out = desc_pre_model([ref_in, style_in])
        loss = pairWiseRankingLoss(ref_out, style_out, label_in)

    #val_metrics.update_state(ref_out, style_out, label_in)
    return loss


def train(epochs=3):
    tensorboard_callback.set_model(desc_pre_model)
    # plotlosses = PlotLosses(outputs=[MatplotlibPlot()], groups={'loss' : ['tr_loss', 'val_loss'], 'accuracy' : ['tr_acc', 'val_acc']})
    for epoch in range(epochs):
        start_time = time.time()
        
        # Iterate over the batches of the dataset.
        pb = tqdm(train_dataset)
        e = 0
        for ref_batch_train, style_batch_train, reflbl_batch_train, stllbl_batch_train in pb:
            pb.set_description(f"[ {epoch}/ {e}] ")
            train_loss = train_step(ref_batch_train, style_batch_train, reflbl_batch_train, stllbl_batch_train)
            #print(f"Epoch {epoch} / step : {step} : loss {train_loss}",end='\r')
            pb.set_postfix(loss=train_loss.numpy())
            e += 1
        # Run a validation loop at the end of each epoch.
        # for ref_batch_val, style_batch_val, label_batch_val in val_dataset:
        #     val_loss = val_step(ref_batch_val, style_batch_val, label_batch_val)

        print(f'Epoch {epoch} : train_loss : {train_loss}')
        # tr_acc = train_metrics.result()
        # val_acc = val_metrics.result()
        # plotlosses.update({
        #     'tr_loss' : train_loss,
        #     'tr_acc' : tr_acc,
        #     'val_loss' : val_loss,
        #     'val_acc' : val_acc,
        # })
        # plotlosses.send()

        # train_metrics.reset_states()
        # val_metrics.reset_states()
        # val_acc = val_metrics.result()
        # val_metrics.reset_states()
        # print("Validation acc: %.4f" % (float(val_acc),))
        print("Time taken: %.2fs" % (time.time() - start_time))

#%%
if __name__ == "__main__":
    #data importing
    stenc_df = pd.read_csv('./data/data/StyleDataset/StyleEnc.csv', index_col=0)
    train_path = pathlib.Path(os.path.join(config.DESC_ROOT_DIR,'train'))
    val_path = pathlib.Path(os.path.join(config.DESC_ROOT_DIR,'validation'))
    #train_gen = gen(1, 2923, './data/data/StyleDataset', 2900)
    #val_gen = gen(2923, 3164, './data/data/StyleDataset', 240)
    train_ds = tf.data.Dataset.from_generator(
        train_gen,
        output_signature=(
            tf.TensorSpec(shape=(128,128, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(128,128,3), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32),
            tf.TensorSpec(shape=(), dtype=tf.int32)
        )

    )
    # val_ds = tf.data.Dataset.from_generator(
    #     val_gen,
    #     output_signature=(
    #         tf.TensorSpec(shape=(128,128, 3), dtype=tf.float32),
    #         tf.TensorSpec(shape=(128,128,3), dtype=tf.float32),
    #         tf.TensorSpec(shape=(), dtype=tf.int32),
    #         tf.TensorSpec(shape=(), dtype=tf.int32)
    #     )

    # )
    #filtered_ds = list_ds.filter(lambda x: int(x.split(os.sep)[-1].strip('.jpg')) < config.DESC_TRAIN_SIZE)
    #sample_dt = sample_ds.shuffle(buffer_size=1000)   #config param
    train_dataset = prepare(train_ds, shuffle=True, augment=True)

    # val_dataset = prepare(val_ds, shuffle=True, augment=False) 
    # init model
    base_model = define_stl_encoder(config.DESCS_LATENT_SIZE, 36, config.IMAGE_SHAPE)

    #train_steps = 100
    #lr_fn = tf.optimizers.schedules.PolynomialDecay(1e-3, train_steps, 1e-5, 2)
    opt = tf.optimizers.Adam(0.001)

    # train_metrics = MarginalAcc()
    # val_metrics = MarginalAcc()
    #desc_pre_model = define_descrminator((config.IMG_WIDTH, config.IMG_HEIGHT, 3))
    desc_pre_model = EmbStyleNet(base_model)

    train(10)
    # tf.profiler.experimental.client.trace('grpc://localhost:6009',
    #                                   config.LOG_DIR+'/profilers', 2000)
    # filename = 'descs_wgt1.h5'
    # base_model.save_weights(os.path.join(config.MODEL_DIR, filename))
    # logger.info(f">> Saved : {filename}  ")

# %%
weights = tf.Variable(base_model.get_layer('embedding').get_weights()[0][1:])
# Create a checkpoint from embedding, the filename and key are the
# name of the tensor.
checkpoint = tf.train.Checkpoint(embedding=weights)
checkpoint.save(os.path.join('./logs/gan', "embedding.ckpt"))

# %%
