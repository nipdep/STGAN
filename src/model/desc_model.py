#! usr/bin/python3

import config
import os
import pathlib
import pandas as pd  
import random
import tensorflow as tf  

root_path = pathlib.Path(config.DESC_ROOT_DIR)
list_ds = tf.data.Dataset.list_files(str(config.DESC_ROOT_DIR/'*.jpg'))
list_ds = list_ds.shuffle(buffer_size=1000)
filtered_ds = list_ds.filter(lambda x: int(bytes.decode(x.numpy()).split(os.sep)[-1].strip('.jpg')) < config.DESC_TRAIN_SIZE)
stenc_df = pd.read_csv(config.DESC_ENC_DIR)['style_code'].tolist()


def process_img(img):
  img = tf.image.decode_jpeg(img, channels=3) 
  img = tf.image.convert_image_dtype(img, tf.float32) 
  rnd_state = random.randint(0,1)
  if rnd_state:
      img = tf.image.random_crop(img, (config.IMG_WIDTH, config.IMG_HEIGHT,3))
  return tf.image.resize(img, config.IMAGE_SIZE)

def process_path(file_path):
    str_file_path = bytes.decode(file_path.numpy())
    cur_ind = str_file_path.split(os.sep)[-1].strip('.jpg')
    cur_style = stenc_df[int(cur_ind)-1]
    random_num = random.randint(0, config.DESC_TRAIN_SIZE)
    random_bool = random.randint(0,1)
    if random_bool:
        if random_num == int(cur_ind):
            random_num = random.randint(0, config.DESC_TRAIN_SIZE)
    else:
        random_num = max(random.randint(0,10), int(cur_ind)-5)
    rand_style = stenc_df[random_num-1]
    cur_img = process_img(tf.io.read_file(file_path))
    rand_path = tf.strings.join([*str_file_path.split(os.sep)[:-1],f'{random_num}.jpg'],os.sep)
    rand_img = process_img(tf.io.read_file(rand_path))

    if cur_style == rand_style:
        label = 1
    else:
        label = 0

    return cur_img, rand_img, label

sample_ds = list_ds.map(lambda x: tf.py_function(process_path, [x], [tf.float32, tf.float32, tf.int32]))
#sample_dt = sample_ds.shuffle(buffer_size=1000)   #config param
batched_dt = sample_ds.batch(config.DESC_BATCH_SIZE)