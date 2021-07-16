# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pathlib 
import os
import pandas as pd  
import numpy as np
import random


# %%
root_dir = '../../../data/data/Desc_dataset/'  #config params
style_enc_file = '../../../data/data/Desc_dataset/style_enc.csv' #config params

# %% [markdown]
# ## rename files by giving style index

# %%
root_path = pathlib.Path('../../../data/data/styleU')
idxs = []
styles = []
style_enc = {}
n = 1
m = 10000
for folder in root_path.glob('*'):
    for file in folder.glob('*'):
        m += 1
        # st_name = ''.join(file.stem.split("_")[:-1])
        # ind = file.stem.split("_")[-1]
        # if st_name not in style_enc:
        #     style_enc[st_name] = n
        #     n+=1
        os.rename(file, os.path.join(file.parent, f'{m}.jpg'))
    # idxs.append(m)
    # styles.append(style_enc[st_name])
# st_df = pd.DataFrame(data={'fname' : idxs, 'style_code' : styles})
# st_df.to_csv(style_enc_file, index=False)

# %% [markdown]
# ## build tf dataset and datapipeline

# %%
import tensorflow as tf  


# %%
root_path = pathlib.Path(root_dir)
list_ds = tf.data.Dataset.list_files(str(root_path/'*.jpg'))
list_ds = list_ds.shuffle(buffer_size=1000)
stenc_df = pd.read_csv(style_enc_file)['style_code'].tolist()
LEN = len(stenc_df)


# %%
LEN 


# %%
list_ds.element_spec


# %%
for f in list_ds.take(5):
    print(f.numpy())


# %%



def process_path(file_path):
  label = tf.strings.split(file_path, os.sep)[-2]
  return tf.io.read_file(file_path)


IMG_WIDTH, IMG_HEIGHT = 128, 128 #config params
def process_img(img):
  img = tf.image.decode_jpeg(img, channels=3) 
  img = tf.image.convert_image_dtype(img, tf.float32) 
  rnd_state = random.randint(0,1)
  if rnd_state:
      img = tf.image.random_crop(img, (IMG_WIDTH, IMG_HEIGHT,3))
  return tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])

def process_path(file_path):
    str_file_path = bytes.decode(file_path.numpy())
    cur_ind = str_file_path.split(os.sep)[-1].strip('.jpg')
    cur_style = stenc_df[int(cur_ind)-1]
    random_num = random.randint(0, LEN)
    random_bool = random.randint(0,1)
    if random_bool:
        if random_num == int(cur_ind):
            random_num = random.randint(0, LEN)
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
batched_dt = sample_ds.batch(16)  #config param

#%%
from matplotlib import pyplot as plt

# %% 
# dataset sampling vis.
num = 10
i = 0
for f in sample_ds.take(num):
    i+=1
    label = f[2].numpy()
    fir_img = f[0].numpy()*255
    sec_img = f[1].numpy()*255
    plt.subplot(2, num, i)
    plt.axis('off')
    plt.title(str(label))
    plt.imshow(fir_img.astype('uint8'))

    plt.subplot(2, num, i+num)
    plt.axis('off')
    plt.title(str(label))
    plt.imshow(sec_img.astype('uint8'))


# %%
dt_path = "../../../data/data/styleU"
style_enc_file = "../../../data/data/styleU/StyleEnc.csv"
root_path = pathlib.Path(dt_path)
idxs = []
styles = []
states = []
paths = []
style_enc = {}
n = 1
m = 0
# for partition in root_path.glob('*'):
#     state = partition.stem[0]
for folder in root_path.glob('*'):
    stl = int(folder.stem)
    for file in folder.glob('*'):
        fpath = '/'.join([*str(file).split('\\')[-2:-1], f'{n}.jpg'])
        os.rename(file, os.path.join(file.parent, f'{n}.jpg'))
        idxs.append(n)
        styles.append(stl)
        #states.append(state)
        paths.append(fpath)
        n+=1

st_df = pd.DataFrame(data={'fname' : idxs, 'path' : paths, 'style_code' : styles})
st_df.to_csv(style_enc_file, index=False)
# %%
import pandas as pd 
# %%
style_enc_file = "../../../data/data/styleU/StyleEnc.csv"
df = pd.read_csv(style_enc_file, index_col=0)
df.shape[0]
# %%
l =df.loc[1,['path', 'style_code']]
# %%
l['path']
# %%
