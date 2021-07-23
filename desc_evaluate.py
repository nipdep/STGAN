#%%
import os
import cv2 
import pathlib
import math
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from numpy import load, vstack, expand_dims
import matplotlib.pyplot as plt
from src.model.cntdesc_model import define_cont_encoder, ContentNet


#%%
model_dir = "data/models/descc_wgt4.h5"
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
    images = (images+1)/2
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

def evaluate(model, dataset, n_samples):
    n = np.random.choice(range(dataset.shape[0]), n_samples, replace=False)
    samples = dataset[n, ...]
    vec1, vec2, vec3 = model(samples[:, 0, ...]), model(samples[:, 1, ...]), model(samples[:, 2, ...])
    dis1 = np.asarray(distance(vec1, vec2))
    dis2 = np.asarray(distance(vec1, vec3))
    for i in range(n_samples):
        plt.subplot(n_samples, 3, 1 + i*3)
        plt.axis('off')
        plt.imshow(rescale(samples[i][0]))
        plt.subplot(n_samples, 3, 2 + i*3)
        plt.axis('off')
        plt.title(str(dis1[i]))
        plt.imshow(rescale(samples[i][1]))
        plt.subplot(n_samples, 3, 3 + i*3)
        plt.axis('off')
        plt.title(str(dis2[i]))
        plt.imshow(rescale(samples[i][2]))
    plt.show()
#%%
if __name__ == '__main__':
    dv_mat = np.swapaxes(np.load('./data/data/desc_validation.npz')['content'], 0, 1)[500:, ...]
    model = define_cont_encoder(32, (128, 128, 3))
    model.load_weights(model_dir)
    n_samples = 4
    evaluate(model, dv_mat, n_samples)
    # cnt_img = img_resize(content_image_dir)
    # style_img = img_resize(style_image_dir)
    # gen_img = generate_image(model, style_img, cnt_img)
    # plot_images(cnt_img, style_img, gen_img)

#%%