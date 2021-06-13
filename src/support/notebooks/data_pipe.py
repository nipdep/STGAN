# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pathlib 
import os
import pandas as pd  
import random


# %%
root_dir = '../../../data/data/Desc_dataset/'
style_enc_file = '../../../data/data/Desc_dataset/style_enc.csv'

# %% [markdown]
# ## rename files by giving style index

# %%
root_path = pathlib.Path(root_dir)
idxs = []
styles = []
style_enc = {}
n = 1
m = 0
for file in root_path.glob('*'):
    m += 1
    st_name = ''.join(file.stem.split("_")[:-1])
    ind = file.stem.split("_")[-1]
    if st_name not in style_enc:
        style_enc[st_name] = n
        n+=1
    os.rename(file, os.path.join(file.parent, f'{m}.jpg'))
    idxs.append(m)
    styles.append(style_enc[st_name])
st_df = pd.DataFrame(data={'fname' : idxs, 'style_code' : styles})
st_df.to_csv(style_enc_file, index=False)

# %% [markdown]
# ## build tf dataset and datapipeline

# %%
import tensorflow as tf  


# %%
root_path = pathlib.Path(root_dir)
list_ds = tf.data.Dataset.list_files(str(root_path/'*.jpg'))
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



# %%
def process_path(file_path):
  label = tf.strings.split(file_path, os.sep)[-2]
  return tf.io.read_file(file_path)


# %%

IMG_WIDTH, IMG_HEIGHT = 32 , 32
def process_img(img):
  img = tf.image.decode_jpeg(img, channels=3) 
  img = tf.image.convert_image_dtype(img, tf.float32) 
  return tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])

def process_path(file_path):
    str_file_path = bytes.decode(file_path.numpy())
    cur_ind = str_file_path.split(os.sep)[-1].strip('.jpg')
    cur_style = stenc_df[int(cur_ind)-1]
    random_num = random.randint(0, LEN)
    if random_num == int(cur_ind):
        random_num = random.randint(0, LEN)
    rand_style = stenc_df[random_num-1]
    cur_img = process_img(tf.io.read_file(file_path))
    rand_path = tf.strings.join([*str_file_path.split(os.sep)[:-1],f'{random_num}.jpg'],os.sep)
    rand_img = process_img(tf.io.read_file(rand_path))

    if cur_style == rand_style:
        label = 1
    else:
        label = 0

    return cur_img, rand_img, label

# def load_audio_file(file_path):
#     # you should decode bytes type to string type
#     print("file_path: ",bytes.decode(file_path.numpy()),type(bytes.decode(file_path.numpy())))
#     return file_path

sample_dt = list_ds.map(lambda x: tf.py_function(process_path, [x], [tf.float32, tf.float32, tf.int32]))
for f in sample_dt.take(500):
    print(f[2])


# %%



