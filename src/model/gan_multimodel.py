#%%
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Reshape, Concatenate, Conv2D, UpSampling2D, BatchNormalization

tfd = tfp.distributions
tfb = tfp.bijectors
tfpl = tfp.layers
#%%

def get_prior(num_modes, latent_dim):
    """
    This function should create an instance of a MixtureSameFamily distribution 
    according to the above specification. 
    The function takes the num_modes and latent_dim as arguments, which should 
    be used to define the distribution.
    Your function should then return the distribution instance.
    """
    prior = tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(probs=[1/num_modes,]*num_modes),
        components_distribution=tfd.MultivariateNormalDiag(
            loc=tf.Variable(tf.random.normal([num_modes, latent_dim])),
            scale_diag=tfp.util.TransformedVariable(tf.Variable(tf.ones([num_modes, latent_dim])), bijector=tfb.Softplus()),
        )
    )
    return prior

prior = get_prior(num_modes=2, latent_dim=50)
#%%

def get_kl_regularizer(prior_distribution):
    """
    This function should create an instance of the KLDivergenceRegularizer 
    according to the above specification. 
    The function takes the prior_distribution, which should be used to define 
    the distribution.
    Your function should then return the KLDivergenceRegularizer instance.
    """
    divergent_regularizer = tfpl.KLDivergenceRegularizer(prior_distribution,
    use_exact_kl=False,
    weight=1.0,
    test_points_fn=lambda t: t.sample(3),
    test_points_reduce_axis=(0,1))
    return divergent_regularizer
kl_regularizer = get_kl_regularizer(prior)

#%%
def get_encoder(latent_dim, kl_regularizer):
    """
    This function should build a CNN encoder model according to the above specification. 
    The function takes latent_dim and kl_regularizer as arguments, which should be
    used to define the model.
    Your function should return the encoder model.
    """
    model = Sequential([
        Conv2D(filters=32,  kernel_size=4, activation='relu', strides=2, padding='SAME', input_shape=(64, 64, 3)),
        BatchNormalization(),
        Conv2D(filters=64,  kernel_size=4, activation='relu', strides=2, padding='SAME'),
        BatchNormalization(),
        Conv2D(filters=128,  kernel_size=4, activation='relu', strides=2, padding='SAME'),
        BatchNormalization(),
        Conv2D(filters=256,  kernel_size=4, activation='relu', strides=2, padding='SAME'),
        BatchNormalization(),
        Flatten(),
        Dense(tfpl.MultivariateNormalTriL.params_size(latent_dim)),
        tfpl.MultivariateNormalTriL(latent_dim, activity_regularizer=kl_regularizer)
    ])
    return model

encoder = get_encoder(latent_dim=50, kl_regularizer=kl_regularizer)

def get_decoder(latent_dim):
    """
    This function should build a CNN decoder model according to the above specification. 
    The function takes latent_dim as an argument, which should be used to define the model.
    Your function should return the decoder model.
    """
    decoder = Sequential([
        Dense(4096, activation='relu', input_shape=(latent_dim,)),
        Reshape((4, 4, 256)),
        UpSampling2D(size=(2, 2)),
        Conv2D(filters=128, kernel_size=3, activation='relu', padding='SAME'),
        UpSampling2D(size=(2, 2)),
        Conv2D(filters=64, kernel_size=3, activation='relu', padding='SAME'),
        UpSampling2D(size=(2, 2)),
        Conv2D(filters=32, kernel_size=3, activation='relu', padding='SAME'),
        UpSampling2D(size=(2, 2)),
        Conv2D(filters=128, kernel_size=3, activation='relu', padding='SAME'),
        Conv2D(filters=3, kernel_size=3, padding='SAME'),
        Flatten(),
        tfpl.IndependentBernoulli(event_shape=(64, 64, 3))
    ])
    return decoder

decoder = get_decoder(latent_dim=50)



#%%

def reconstruction_loss(batch_of_images, decoding_dist):
    """
    This function should compute and return the average expected reconstruction loss,
    as defined above.
    The function takes batch_of_images (Tensor containing a batch of input images to
    the encoder) and decoding_dist (output distribution of decoder after passing the 
    image batch through the encoder and decoder) as arguments.
    The function should return the scalar average expected reconstruction loss.
    """
    return -tf.reduce_mean(decoding_dist.log_prob(batch_of_images), axis=0)
    
#%%
