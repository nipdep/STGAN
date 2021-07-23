
#%%

import os
import random
import math
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow_addons.layers import InstanceNormalization
#%%

def pixel_norm(x, epsilon=1e-8):
    return x / tf.math.sqrt(tf.reduce_mean(x ** 2, axis=-1, keepdims=True) + epsilon)

class EqualizedDense(layers.Layer):
    def __init__(self, units, gain=2, learning_rate_multiplier=1, **kwargs):
        super(EqualizedDense, self).__init__(**kwargs)
        self.units = units
        self.gain = gain
        self.learning_rate_multiplier = learning_rate_multiplier

    def build(self, input_shape):
        self.in_channels = input_shape[-1]
        initializer = keras.initializers.RandomNormal(
            mean=0.0, stddev=1.0 / self.learning_rate_multiplier
        )
        self.w = self.add_weight(
            shape=[self.in_channels, self.units],
            initializer=initializer,
            trainable=True,
            name="kernel",
        )
        self.b = self.add_weight(
            shape=(self.units,), initializer="zeros", trainable=True, name="bias"
        )
        fan_in = self.in_channels
        self.scale = tf.sqrt(self.gain / fan_in)

    def call(self, inputs):
        output = tf.add(tf.matmul(inputs, self.scale * self.w), self.b)
        return output * self.learning_rate_multiplier

def mapping_model(num_layers, num_classes, latent_size, input_shape=(128, 128, 3)):
    #z = layers.Input(shape=(input_shape))
    feat_model = tf.keras.applications.densenet.DenseNet121(weights='imagenet', include_top=False, input_shape=input_shape)
    for layer in feat_model.layers:
        layer.trainable = False 
    w = feat_model.get_layer("conv4_block9_0_relu").output
    w = layers.GlobalMaxPooling2D()(w)
    w = EqualizedDense(256, learning_rate_multiplier=0.01)(w)
    w = layers.LeakyReLU(0.2)(w)
    w = EqualizedDense(128, learning_rate_multiplier=0.01)(w)
    w = layers.LeakyReLU(0.2)(w)
    w = layers.Dense(latent_size, activation='relu')(w)
    output = layers.Dense(num_classes, activation='softmax', name='stl_output')(w)
    model = keras.Model(inputs=feat_model.input, outputs=output, name='style_mapping')
    return model

#%%
# model = mapping_model(2, 12, 64)
# tf.keras.utils.plot_model(model, show_shapes=True)
# model.summary()
# %%
