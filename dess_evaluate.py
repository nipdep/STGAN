#%%
import os
import cv2 
import pathlib
import math
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import tensorflow.keras.layers.experimental.preprocessing as prep
from numpy import load, vstack, expand_dims
import matplotlib.pyplot as plt

import config
from src.model.stldesc_model import define_desc_encoder, StyleNet, define_stl_encoder

#%%
model_dir = "data/models/descs_wgt1.h5"
style_image_dir = "data/data/styles/the_scream.jpg"
content_image_dir = "data/data/styles/sample.jpg"

def img_resize(img_path, shape=(128, 128)):
    img = cv2.imread(img_path)
    orig_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return load_image(cv2.resize(orig_img, shape, interpolation=cv2.INTER_LANCZOS4))

def generate_image(model, sty_img, cnt_img):
    gen_img = model([cnt_img, sty_img])
    return gen_img

def distance(embeddings1, embeddings2, distance_metric=0):
    if distance_metric==0:
        # Euclidian distance
        diff = tf.math.subtract(embeddings1, embeddings2)
        dist = tf.reduce_sum(tf.math.pow(diff, 2),1)
    elif distance_metric==1:
        # Distance based on cosine similarity
        dot = tf.math.sum(tf.math.multiply(embeddings1, embeddings2), axis=1)
        norm = tf.math.norm(embeddings1, axis=1) * tf.math.norm(embeddings2, axis=1)
        similarity = dot / norm
        dist = np.arccos(similarity) / math.pi
    else:
        raise 'Undefined distance metric %d' % distance_metric 
        
    return tf.cast(dist, dtype=tf.float32)

def load_image(img):
    rc_pixels = (img-127.5)/127.5
    pixels = expand_dims(rc_pixels, axis=0)
    return pixels

def plot_images(cnt_img, style_img, gen_img):
    images = vstack((cnt_img, style_img, gen_img))
    images = (images+1)/2.0
    titles = ['Content image', 'Style image', 'Generated image']

    for i in range(len(images)):
        plt.subplot(1, 3, 1+i)
        plt.axis('off')
        plt.imshow(images[i])
        plt.title(titles[i])
    plt.show()
    plt.savefig("sample.jpg")
    plt.close()

def rescale(img):
    return (img+1)/2

def predict(model, x):
    y = model(x)
    return np.asarray(y)

def process_path(file_path):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    #print(fp)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, size=(128, 128))
    return img

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

        label = 0
        if img1_det['style_code'] == img2_det['style_code']:
            label = 1
        #print(os.path.join(root_path, img1_det['path']), os.path.join(root_path, img2_det['path']))
        try :
            img1 = process_path(os.path.join(root_path, img1_det['path']))
            img2 = process_path(os.path.join(root_path, img2_det['path']))
            yield img1, img2, label
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

    ds = ds.map(lambda x1, x2, y: (resize_and_rescale(x1), resize_and_rescale(x2), y),
                num_parallel_calls=tf.data.AUTOTUNE)

    #ds = ds.cache()
    
    if shuffle:
        ds = ds.shuffle(1000)

    ds = ds.batch(16)

    if augment:
        ds = ds.map(lambda x1, x2, y: (data_augmentation(x1, training=True), data_augmentation(x2, training=True), y), 
                    num_parallel_calls=tf.data.AUTOTUNE)
    
    return ds.prefetch(buffer_size=tf.data.AUTOTUNE)

def evaluate(model, dataset, n_samples):
    #n = np.random.choice(range(dataset.shape[0]), n_samples, replace=False)
    #samples = dataset[n, ...]
    for img1, img2, labels in dataset:
        vec1, vec2, lbl= model(img1), model(img2), labels
        break
    #vec1, vec2= model(img1), model(img2)
    dis1 = np.asarray(distance(vec1, vec2))
    #dis2 = np.asarray(distance(vec1, vec3))
    plt.figure(figsize=(4, 20))
    for i in range(n_samples):
        plt.subplot(n_samples, 2, 1 + i*2)
        plt.axis('off')
        plt.imshow(rescale(img1[i]))
        plt.subplot(n_samples, 2, 2 + i*2)
        plt.axis('off')
        plt.title(str(dis1[i])+" | "+str(lbl[i].numpy()))
        plt.imshow(rescale(img2[i]))
        #plt.subplot(n_samples, 2, 3 + i*3)
        # plt.axis('off')
        # plt.title(str(dis2[i]))
        # plt.imshow(rescale(samples[i][2]))
    plt.show()

if __name__ == '__main__':
    stenc_df = pd.read_csv('./data/data/StyleDataset/StyleEnc.csv', index_col=0)
    val_path = pathlib.Path(os.path.join(config.DESC_ROOT_DIR,'validation'))
    val_ds = tf.data.Dataset.from_generator(
        val_gen,
        output_signature=(
            tf.TensorSpec(shape=(128,128, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(128,128,3), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32),
        )

    )
    val_dataset = prepare(val_ds, shuffle=True, augment=False)
    model = define_stl_encoder(config.DESCS_LATENT_SIZE, config.IMAGE_SHAPE)
    model.load_weights(model_dir)
    n_samples = 10
    evaluate(model, val_dataset, n_samples)
    # cnt_img = img_resize(content_image_dir)
    # style_img = img_resize(style_image_dir)
    # gen_img = generate_image(model, style_img, cnt_img)
    # plot_images(cnt_img, style_img, gen_img)

#%%