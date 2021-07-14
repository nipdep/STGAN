#%%
import os
import cv2 
import pathlib
import tensorflow as tf
from tensorflow.keras.models import load_model
from numpy import load, vstack, expand_dims
import matplotlib.pyplot as plt

#%%
model_dir = "data/models/gen_model.h5"
style_image_dir = "data/data/styles/the_scream.jpg"
content_image_dir = "data/data/styles/sample.jpg"

def img_resize(img_path, shape=(128, 128)):
    img = cv2.imread(img_path)
    orig_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return load_image(cv2.resize(orig_img, shape, interpolation=cv2.INTER_LANCZOS4))

def generate_image(model, sty_img, cnt_img):
    gen_img = model([cnt_img, sty_img])
    return gen_img

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

if __name__ == '__main__':
    model = load_model(model_dir)
    cnt_img = img_resize(content_image_dir)
    style_img = img_resize(style_image_dir)
    gen_img = generate_image(model, style_img, cnt_img)
    plot_images(cnt_img, style_img, gen_img)

#%%