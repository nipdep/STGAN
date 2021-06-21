
# %%
import time
import tensorflow as tf
from numpy import load
from numpy import zeros
from numpy import ones
from numpy.random import randint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Conv2DTranspose, LeakyReLU, Activation, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras import losses
from tensorflow.keras import metrics 
from matplotlib import pyplot

#%%

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
    return output
#%%

desc_pre_model = define_descrminator((128, 128, 3))
desc_pre_model.summary()

# %%

tf.keras.utils.plot_model(desc_pre_model, show_shapes=True)
# %%
##build the model

epochs = 10
opt = Adam(lr=0.002)
desc_loss = losses.Hinge()
train_metrics = metrics.Hinge()
val_metrics = metrics.Hinge()


@tf.function
def train_step(ref_in, style_in):
    with tf.GradientTape() as tape:
        ref_out = desc_pre_model(ref_in)
        style_out = desc_pre_model(style_in)
        loss = desc_loss(ref_out, style_out)
    grads = tape.gradient(loss, desc_pre_model.trainable_weights)
    opt.apply_gradients(zip(grads, desc_pre_model.trainable_weights))
    train_metrics.update(ref_out, style_out)
    return loss

@tf.function
def val_step(ref_in, style_in):
    ref_out = desc_pre_model(ref_in)
    style_out = desc_pre_model(style_in)
    val_metrics.update(ref_out, style_out)

for epoch in range(epochs):
    print("\nStart of epoch %d" % (epoch,))
    start_time = time.time()

    # Iterate over the batches of the dataset.
    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
        loss_value = train_step(x_batch_train, y_batch_train)

        # Log every 200 batches.
        if step % 200 == 0:
            print(
                "Training loss (for one batch) at step %d: %.4f"
                % (step, float(loss_value))
            )
            print("Seen so far: %d samples" % ((step + 1) * 64))

    # Display metrics at the end of each epoch.
    train_acc = train_metrics.result()
    print("Training acc over epoch: %.4f" % (float(train_acc),))

    # Reset training metrics at the end of each epoch
    train_metrics.reset_states()

    # Run a validation loop at the end of each epoch.
    # for x_batch_val, y_batch_val in val_dataset:
    #     val_step(x_batch_val, y_batch_val)

    # val_acc = val_metrics.result()
    # val_metrics.reset_states()
    # print("Validation acc: %.4f" % (float(val_acc),))
    print("Time taken: %.2fs" % (time.time() - start_time))



#%%
