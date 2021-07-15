#%%
import tensorflow as tf
import tensorflow_addons as tfa
from datetime import datetime

import config
from src.model.stldesc_model import define_desc_encoder, StyleNet, define_stl_classifier

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Hinge
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Model


#%%

bt = 16

datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.4,
    height_shift_range=0.5,
    brightness_range=(0.4, 0.7),
    shear_range=0.2,
    zoom_range=(0.5,0.5),
    horizontal_flip=True,
    vertical_flip=True,
    rescale=1/255.,
    validation_split=0.1
)

train_dt = datagen.flow_from_directory(
    config.DESC_STL_DIR,
    target_size=(128, 128),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=bt,
    shuffle=True,
    seed=114,
    subset='training',
    interpolation='lanczos'
)

val_dt = datagen.flow_from_directory(
    config.DESC_STL_DIR,
    target_size=(128, 128),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=bt,
    shuffle=False,
    seed=114,
    subset='validation',
    interpolation='lanczos'
)

STEP_SIZE_TRAIN=train_dt.n//train_dt.batch_size
STEP_SIZE_VALID=val_dt.n//val_dt.batch_size

#%%

base_model = define_stl_classifier(config.DESCS_LATENT_SIZE, 36,  config.IMAGE_SHAPE)
base_model.compile(
    optimizer=Adam(0.002),
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
    metrics=[tf.keras.metrics.CategoricalAccuracy()]
)


# %%

history = base_model.fit(
    train_dt,
    batch_size=bt,
    epochs=30,
    steps_per_epoch=STEP_SIZE_TRAIN,
    validation_data=val_dt,
    validation_steps=STEP_SIZE_VALID,
    workers=tf.data.AUTOTUNE,
)

#%%

base_model.save('./data/models/dess_m1.h5')
# %%
