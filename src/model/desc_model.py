#! usr/bin/python3
#%%
%cd ../../
import config
import os
import logging
import time
from datetime import datetime
import pathlib
import pandas as pd  
import random
import tensorflow as tf  
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Conv2DTranspose, LeakyReLU, Activation, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras import losses
from tensorflow.keras import metrics 
from tensorflow.keras.callbacks import TensorBoard
#%%
#tensorboard logger
logdir = config.LOG_DIR+ "/desc_pre_" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=logdir, histogram_freq=1, profile_batch=1)

tf.profiler.experimental.server.start(6009)

# set logger
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

root_path = pathlib.Path(config.DESC_ROOT_DIR)
list_ds = tf.data.Dataset.list_files(str(root_path/'*.jpg'))
list_ds = list_ds.shuffle(buffer_size=1000)
#filtered_ds = list_ds.filter(lambda x: int(x.split(os.sep)[-1].strip('.jpg')) < config.DESC_TRAIN_SIZE)
stenc_df = pd.read_csv(config.DESC_ENC_DIR)['style_code'].tolist()

def process_img(img):
  img = tf.image.decode_jpeg(img, channels=3) 
  img = tf.image.convert_image_dtype(img, tf.float32) 
  rnd_state = random.randint(0,1)
  if rnd_state:
        try:
            img = tf.image.random_crop(img, (config.IMG_WIDTH, config.IMG_HEIGHT,3))
        except :
            logger.warning("image shape is less than the cropping size.")
  return tf.image.resize(img, config.IMAGE_SIZE)

def process_path(file_path):
    str_file_path = bytes.decode(file_path.numpy())
    cur_ind = str_file_path.split(os.sep)[-1].strip('.jpg')
    cur_style = stenc_df[int(cur_ind)-1]
    random_num = random.randint(1, config.DESC_TRAIN_SIZE)
    random_bool = random.randint(0,1)
    if random_bool:
        if random_num == int(cur_ind):
            random_num = random.randint(1, config.DESC_TRAIN_SIZE)
    else:
        random_num = max(random.randint(1,10), int(cur_ind)-5)
    rand_style = stenc_df[random_num-1]
    cur_img = process_img(tf.io.read_file(file_path))
    rand_path = tf.strings.join([*str_file_path.split(os.sep)[:-1],f'{random_num}.jpg'],os.sep)
    rand_img = process_img(tf.io.read_file(rand_path))

    if cur_style == rand_style:
        label = 1
    else:
        label = 0

    return cur_img, rand_img, label

sample_ds = list_ds.map(lambda x: tf.py_function(process_path, [x], [tf.float32, tf.float32, tf.int32]),
                        num_parallel_calls=tf.data.AUTOTUNE,
                        deterministic=False)
#sample_dt = sample_ds.shuffle(buffer_size=1000)   #config param
batched_dt = sample_ds.batch(batch_size=config.DESC_BATCH_SIZE)
train_dataset = batched_dt.cache().prefetch(tf.data.AUTOTUNE)

# TODO://layers and width parameter tune
def define_descrminator(image_size):
    init = RandomNormal(stddev=0.02)
    input_img = Input(shape=image_size)
    # C64
    d = Conv2D(64, (4, 4), (2, 2), padding='SAME', kernel_initializer=init)(input_img)
    d = LeakyReLU(alpha=0.2)(d)
	# C128
    d = Conv2D(128, (4, 4), (2, 2), padding='SAME', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
	# C256
    d = Conv2D(256, (4, 4), (2, 2), padding='SAME', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # flatten
    flt = Flatten()(d)
    # linear logits layer
    output = Dense(1)(flt)
    #build and compile the model
    model = Model(inputs=input_img, outputs=output, name='style_descriminator')
    return model

epochs = config.DESC_EPOCHS
opt = Adam(lr=config.DESC_INIT_LR)
desc_loss = losses.Hinge()
train_metrics = metrics.Hinge()
val_metrics = metrics.Hinge()
desc_pre_model = define_descrminator((config.IMG_WIDTH, config.IMG_HEIGHT, 3))

def pairWiseRankingLoss(y_ref, y_style, label):
    m  = tf.cast(tf.broadcast_to(config.M, shape=y_ref.shape), dtype=tf.float32)
    u  = tf.cast(tf.broadcast_to(0, shape=y_ref.shape), dtype=tf.float32)
    i  = tf.cast(tf.broadcast_to(1, shape=y_ref.shape), dtype=tf.float32)
    y = tf.cast(label[..., tf.newaxis], dtype=tf.float32)
    dist = tf.norm(y_ref-y_style, ord='euclidean', axis=-1, keepdims=True)
    loss = y*dist + (i-y)*tf.reduce_max(tf.stack([u,m-dist]), axis=0)
    return tf.reduce_mean(loss)

@tf.function
def train_step(ref_in, style_in, label_in):
    with tf.GradientTape() as tape:
        ref_out = desc_pre_model(ref_in)
        style_out = desc_pre_model(style_in)
        loss = pairWiseRankingLoss(ref_out, style_out, label_in)
    grads = tape.gradient(loss, desc_pre_model.trainable_weights)
    opt.apply_gradients(zip(grads, desc_pre_model.trainable_weights))
    #train_metrics.update(ref_out, style_out)
    return loss

@tf.function
def val_step(ref_in, style_in, label_in):
    ref_out = desc_pre_model(ref_in)
    style_out = desc_pre_model(style_in)
    val_metrics.update(ref_out, style_out)

tensorboard_callback.set_model(desc_pre_model)

for epoch in range(epochs):
    print("\nStart of epoch %d" % (epoch,))
    start_time = time.time()
    
    # Iterate over the batches of the dataset.
    for step, (ref_batch_train, style_batch_train, label_batch_train) in enumerate(train_dataset):
        loss_value = train_step(ref_batch_train, style_batch_train, label_batch_train)

        # Log every 200 batches.
        if step % 200 == 0:
            print(
                "Training loss (for one batch) at step %d: %.4f"
                % (step, float(loss_value))
            )
            print("Seen so far: %d samples" % ((step + 1) * 64))

    # Display metrics at the end of each epoch.
    # train_acc = train_metrics.result()
    # print("Training acc over epoch: %.4f" % (float(train_acc),))

    # # Reset training metrics at the end of each epoch
    # train_metrics.reset_states()

    # Run a validation loop at the end of each epoch.
    # for x_batch_val, y_batch_val in val_dataset:
    #     val_step(x_batch_val, y_batch_val)

    # val_acc = val_metrics.result()
    # val_metrics.reset_states()
    # print("Validation acc: %.4f" % (float(val_acc),))
    print("Time taken: %.2fs" % (time.time() - start_time))

tf.profiler.experimental.client.trace('grpc://localhost:6009',
                                      config.LOG_DIR+'/profilers', 2000)

# %%
