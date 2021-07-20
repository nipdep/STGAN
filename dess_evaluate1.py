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
model_dir = "./data/models/dess_m006.h5"
style_image_dir = "data/data/styles/the_scream.jpg"
content_image_dir = "data/data/styles/sample.jpg"

def img_resize(img_path, shape=(128, 128)):
    img = cv2.imread(img_path)
    orig_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(orig_img, shape, interpolation=cv2.INTER_LANCZOS4)
    return img

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

def plot_result(images, y, preds):
    images = (images+1)/2.0
    l = images.shape[0]
    for i in range(l):
        plt.subplot(l, 1, 1+i)
        plt.axis('off')
        plt.imshow(images[i])
        plt.title(f'pred:{round(preds[i].numpy()[0])}|y:{y[i]}')
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

def evaluate(df, n_samples, root, model):
    idx = np.random.choice(range(1, df.shape[0]), n_samples, replace=False)
    t = 0
    y = []
    for ind in idx:
        pixels = img_resize(os.path.join(root,df.loc[ind, 'path']))
        y.append(df.loc[ind, 'style_code'])
        pixels = load_image(pixels)
        if t != 0:
            samples = np.vstack((samples,pixels))
        else:
            samples = pixels
            t +=1
    samples = np.squeeze(samples)
    preds = model(samples)
    plot_result(samples, y, preds)

#%%
if __name__ == '__main__':
    stenc_df = pd.read_csv('./data/data/style datasetU/StyleEnc.csv', index_col=0)
    model = load_model(model_dir)
    evaluate(stenc_df, 5, './data/data/style datasetU/data', model)
    # cnt_img = img_resize(content_image_dir)
    # style_img = img_resize(style_image_dir)
    # gen_img = generate_image(model, style_img, cnt_img)
    # plot_images(cnt_img, style_img, gen_img)

#%%


# %%
