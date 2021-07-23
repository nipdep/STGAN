#! usr/bin/python3
#%%
import config
from src.model.stldesc_model import define_desc_encoder, StyleNet, define_stl_encoder, stl_encoder
from src.support.loss_functions import pairWiseRankingLoss, MarginalAcc, triplet_loss


import os
import logging
import time
import math
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
    root_path, n ='./data/data/style datasetU/data', 2900
    idx = np.array(range(n))
    class_id = 1
    for i in idx:
        #i = random.randint(lower, higher)
        random_nums = np.random.choice(class_idx_list[class_id-1], 2, replace=False)
        # random_bool = random.randint(0,1)
        # if random_bool:
        #     if random_num == int(i):
        #         random_num = random.randint(lower, higher)
        # else:
        #     random_num = max(random.randint(1,10), int(i)-5)
        img1_det = stenc_df.loc[random_nums[0]-1, ['path', 'style_code']]
        img2_det = stenc_df.loc[random_nums[1]-1, ['path', 'style_code']]
        # label = 0
        # if img1_det['style_code'] == img2_det['style_code']:
        #     label = 1
        #print(os.path.join(root_path, img1_det['path']), os.path.join(root_path, img2_det['path']))
        try :
            img1 = process_path(os.path.join(root_path, img1_det['path']))
            img2 = process_path(os.path.join(root_path, img2_det['path']))
            if class_id < n_classes:
                class_id += 1
            else:
                class_id = 1
            yield img1, img2
        except:
            print(f"Error in file {img1_det['path']}")
            continue

def val_gen():
    root_path, n ='./data/data/style datasetU/data', 200
    idx = np.array(range(n))
    class_id = 1
    for i in idx:
        #i = random.randint(lower, higher)
        random_nums = np.random.choice(class_idx_list[class_id-1], 2, replace=False)
        # random_bool = random.randint(0,1)
        # if random_bool:
        #     if random_num == int(i):
        #         random_num = random.randint(lower, higher)
        # else:
        #     random_num = max(random.randint(1,10), int(i)-5)
        img1_det = stenc_df.loc[random_nums[0]-1, ['path', 'style_code']]
        img2_det = stenc_df.loc[random_nums[1]-1, ['path', 'style_code']]
        # label = 0
        # if img1_det['style_code'] == img2_det['style_code']:
        #     label = 1
        #print(os.path.join(root_path, img1_det['path']), os.path.join(root_path, img2_det['path']))
        try :
            img1 = process_path(os.path.join(root_path, img1_det['path']))
            img2 = process_path(os.path.join(root_path, img2_det['path']))
            if class_id < 12:
                class_id += 1
            else:
                class_id = 1
            yield img1, img2
        except:
            print(f"Error in file {img1_det['path']}")
            continue

# image resize and rescale pipeline
resize_and_rescale = tf.keras.Sequential([
    prep.Resizing(config.IMG_HEIGHT, config.IMG_WIDTH),
    prep.Normalization()
])

# image augmentation pipeline
data_augmentation = tf.keras.Sequential([
    prep.RandomContrast(0.2),
    prep.RandomFlip("horizontal"),
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

    ds = ds.map(lambda x1, x2: (resize_and_rescale(x1), resize_and_rescale(x2)),
                num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.cache()

    ds = ds.batch(12, drop_remainder=True)

    if shuffle:
        ds = ds.shuffle(1000)

    if augment:
        ds = ds.map(lambda x1, x2: (data_augmentation(x1, training=True), data_augmentation(x2, training=True)), 
                    num_parallel_calls=tf.data.AUTOTUNE)
    
    return ds.prefetch(buffer_size=tf.data.AUTOTUNE)

@tf.function
def train_step(ref_in, style_in):
    with tf.GradientTape() as tape:
        ref_emb, pos_emb = desc_pre_model([ref_in, style_in])

        similarity = tf.einsum(
            "ae,pe->ap", ref_emb, pos_emb
        )
        temp = 0.2
        similarity /= temp

        sparse_labels = tf.range(n_classes)

        loss = stlLoss(sparse_labels, similarity)
    grads = tape.gradient(loss, base_model.trainable_variables)
    opt.apply_gradients(zip(grads, base_model.trainable_variables))
    train_metrics.update_state(sparse_labels, similarity)
    return loss

@tf.function
def val_step(ref_in, style_in):
    with tf.GradientTape() as tape:
        ref_emb, pos_emb = desc_pre_model([ref_in, style_in])

        similarity = tf.einsum(
            "ae,pe->ap", ref_emb, pos_emb
        )
        temp = 0.2
        similarity /= temp

        sparse_labels = tf.range(n_classes)
        loss = stlLoss(sparse_labels, similarity)

    val_metrics.update_state(sparse_labels, similarity)
    return loss


def train(epochs=3):
    tensorboard_callback.set_model(desc_pre_model)
    plotlosses = PlotLosses(outputs=[MatplotlibPlot()], groups={'loss' : ['tr_loss', 'val_loss'], 'accuracy' : ['tr_acc', 'val_acc']})
    for epoch in range(epochs):
        start_time = time.time()
        
        # Iterate over the batches of the dataset.
        for step, (ref_batch_train, style_batch_train) in enumerate(train_dataset):
            train_loss = train_step(ref_batch_train, style_batch_train)

        # Run a validation loop at the end of each epoch.
        for ref_batch_val, style_batch_val in val_dataset:
            val_loss = val_step(ref_batch_val, style_batch_val)

        print(f'train_loss : {train_loss} | val_loss : {val_loss}')
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
        # val_acc = val_metrics.result()
        # val_metrics.reset_states()
        # print("Validation acc: %.4f" % (float(val_acc),))
        print("Time taken: %.2fs" % (time.time() - start_time))

#%%
if __name__ == "__main__":
    #data importing
    stenc_df = pd.read_csv('./data/data/style datasetU/StyleEnc.csv')
    train_path = pathlib.Path(os.path.join(config.DESC_ROOT_DIR,'train'))
    val_path = pathlib.Path(os.path.join(config.DESC_ROOT_DIR,'validation'))
    n_classes = 12
    class_idx_list = [np.squeeze(stenc_df.loc[stenc_df['style_code']==i,['fname']].values) for i in range(1, n_classes+1)]
    #train_gen = gen(1, 2923, './data/data/StyleDataset', 2900)
    #val_gen = gen(2923, 3164, './data/data/StyleDataset', 240)
    train_ds = tf.data.Dataset.from_generator(
        train_gen,
        output_signature=(
            tf.TensorSpec(shape=(128,128, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(128,128,3), dtype=tf.float32)
        )

    )
    val_ds = tf.data.Dataset.from_generator(
        val_gen,
        output_signature=(
            tf.TensorSpec(shape=(128,128, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(128,128,3), dtype=tf.float32)
        )

    )
    #filtered_ds = list_ds.filter(lambda x: int(x.split(os.sep)[-1].strip('.jpg')) < config.DESC_TRAIN_SIZE)
    #sample_dt = sample_ds.shuffle(buffer_size=1000)   #config param
    train_dataset = prepare(train_ds, shuffle=True, augment=False)

    val_dataset = prepare(val_ds, shuffle=True, augment=False) 
    # init model
    base_model = stl_encoder(config.DESCS_LATENT_SIZE, config.IMAGE_SHAPE)

    train_steps = 100
    lr_fn = tf.optimizers.schedules.PolynomialDecay(1e-4, train_steps, 1e-5, 2)
    opt = tf.optimizers.Adam(1e-3)
    stlLoss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    train_metrics = tf.keras.metrics.SparseCategoricalCrossentropy(from_logits=True)
    val_metrics = tf.keras.metrics.SparseCategoricalCrossentropy(from_logits=True)
    #desc_pre_model = define_descrminator((config.IMG_WIDTH, config.IMG_HEIGHT, 3))
    desc_pre_model = StyleNet(base_model)
    
    train(config.DESCS_EPOCHS)
    # tf.profiler.experimental.client.trace('grpc://localhost:6009',
    #         
#%%                          config.LOG_DIR+'/profilers', 2000)
filename = 'descs_wgt7.h5'
base_model.save_weights(os.path.join(config.MODEL_DIR, filename))
logger.info(f">> Saved : {filename}  ")

# %%
import io

results = base_model.predict(val_dataset)
np.savetxt("./logs/gan/triplet/vecs4.tsv", results, delimiter='\t')
#%%
out_m = io.open('./logs/gan/triplet/meta4.tsv', 'w', encoding='utf-8')
t = True
labels = []
batchs = []
for img, label in val_dataset:
    # l = ' '.join(map(str, label))
    # if l not in batchs:
    labels.extend(range(12))
    #     batchs.append(l)
    # else:
    #     break
labels.sort()
[out_m.write(str(x) + "\n") for x in labels]
out_m.close()
# %%
near_neighbours_per_example = 12

embeddings = base_model.predict(val_dataset)
gram_matrix = np.einsum("ae,be->ab", embeddings, embeddings)
near_neighbours = np.argsort(gram_matrix.T)[:, -(near_neighbours_per_example + 1) :]
# %%
from sklearn.metrics import ConfusionMatrixDisplay
num_classes = 12
confusion_matrix = np.zeros((num_classes, num_classes))

# For each class.
for class_idx in range(num_classes):
    # Consider 10 examples.
    example_idxs = class_idx_list[class_idx][:12]
    for y_test_idx in example_idxs:
        # And count the classes of its near neighbours.
        for nn_idx in near_neighbours[y_test_idx][:-1]:
            nn_class_idx = labels[nn_idx]
            confusion_matrix[class_idx, nn_class_idx] += 1

# Display a confusion matrix.
labels = range(1, 13)
disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=labels)
disp.plot(include_values=True, cmap="viridis", ax=None, xticks_rotation="vertical")
plt.show()
# %%
