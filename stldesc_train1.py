#%%
import tensorflow as tf
import numpy as np
import tensorflow_addons as tfa
from datetime import datetime

import config
from src.model.stldesc_model import define_desc_encoder, StyleNet, define_stl_regressor, stl_encoder

from tensorflow.keras.optimizers import Adam, Adamax, SGD
from tensorflow.keras.losses import Hinge
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Model

#%%
bt = 32

def prep_fn(img):
    img = img.astype(np.float32) / 255.0
    img = (img - 0.5) * 2
    return img

datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    brightness_range=(0.2, 0.2),
    #shear_range=0.2,
    zoom_range=(0.1,0.1),
    horizontal_flip=True,
    vertical_flip=False,
    #rescale=1/255.,
    preprocessing_function=tf.keras.applications.densenet.preprocess_input,
    validation_split=0.1
)

train_dt = datagen.flow_from_directory(
    config.DESC_STL_DIR,
    target_size=(128, 128),
    color_mode='rgb',
    class_mode='sparse',
    batch_size=bt,
    shuffle=True,
    seed=114,
    subset='training'
)

val_dt = datagen.flow_from_directory(
    config.DESC_STL_DIR,
    target_size=(128, 128),
    color_mode='rgb',
    class_mode='sparse',
    batch_size=bt,
    shuffle=False,
    seed=114,
    subset='validation'
)

STEP_SIZE_TRAIN=train_dt.n//train_dt.batch_size
STEP_SIZE_VALID=val_dt.n//val_dt.batch_size
#%%
lr_fn = tf.optimizers.schedules.PolynomialDecay(1e-4, 150, 1e-5, 2)
base_model = stl_encoder(config.DESCS_LATENT_SIZE, config.IMAGE_SHAPE)
base_model.compile(
    optimizer=Adam(lr_fn),
    loss=tfa.losses.TripletSemiHardLoss(margin=0.7)
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

base_model.save_weights('./data/models/dess_wgt6.h5')
# %%
from matplotlib import pyplot as plt
#%%
plt.plot(history.history['mean_absolute_error'])
plt.plot(history.history['val_mean_absolute_error'])
plt.title('model accuracy')
plt.ylabel('mean absolute error')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
# %%
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()
# %%
import io

results = base_model.predict(val_dt)
np.savetxt("./logs/gan/triplet/vecs1.tsv", results, delimiter='\t')
#%%
out_m = io.open('./logs/gan/triplet/meta1.tsv', 'w', encoding='utf-8')
t = True
labels = []
batchs = []
for img, label in val_dt:
    l = ' '.join(map(str, label))
    if l not in batchs:
        labels.extend(label)
        batchs.append(l)
    else:
        break
labels.sort()
[out_m.write(str(x) + "\n") for x in labels]
out_m.close()
# %%

# %%

# %%

# %%
