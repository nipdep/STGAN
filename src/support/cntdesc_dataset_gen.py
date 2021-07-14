#%%
import numpy as np
from matplotlib import pyplot
import config 
#%%
def load_pixel_metrics(filename):
    full_mat = np.load(filename)
    style_pixels = (full_mat['style']-127.5)/127.5
    content_pixels = (full_mat['cotent']-127.5)/127.5
    transfer_mat  = (full_mat['transfers']-127.5)/127.5
    return style_pixels, content_pixels, transfer_mat

dataset_lis = load_pixel_metrics(config.GAN_DATASET_DIR)
#%%

def pick(x):
    img_set = np.squeeze(full_dt[x])
    l = img_set.shape[0]
    idx =np.random.randint(0, 7)
    return img_set[idx]

def get_shape(x):
    return full_dt[x].shape

f1 = np.vectorize(pick)

def get_rnd_styles(vec, seed=434):
    np.random.seed(seed)
    shf_dt = np.apply_along_axis(pick, 0, vec)
    #shf_dt = f1(vec)
    return shf_dt

full_dt = np.swapaxes(np.vstack([dataset_lis[1][np.newaxis, ...], dataset_lis[2]]), 0, 1)
n = range(full_dt.shape[0])

#shf1_dt = full_dt[idx1]
idx1 = np.random.choice(n, full_dt.shape[0]//2, replace=False)[np.newaxis, ...]
fs1 = np.moveaxis(get_rnd_styles(idx1, 313), [2 ,3], [-1, 0])[:, np.newaxis, ...]
fs2 = np.moveaxis(get_rnd_styles(idx1, 614), [2 ,3], [-1, 0])[:, np.newaxis, ...]
fs3 = np.take(full_dt, 0, 1)[:full_dt.shape[0]//2, np.newaxis, ...]
f_dt = np.hstack([fs1, fs2, fs3])

# idx2 = np.random.choice(n, full_dt.shape[0], replace=False)[np.newaxis, ...]
# ss1 = np.moveaxis(get_rnd_styles(idx2, 115), [2 ,3], [-1, 0])[:, np.newaxis, ...]
# ss2 = np.moveaxis(get_rnd_styles(idx2, 116), [2 ,3], [-1, 0])[:, np.newaxis, ...]
# ss3 = np.moveaxis(get_rnd_styles(idx1, 117), [2 ,3], [-1, 0])[:, np.newaxis, ...]
# s_dt = np.hstack([ss1, ss2, ss3])

# comp_dt = np.vstack([f_dt, s_dt])
#%%
sw_comp_dt = np.swapaxes(f_dt, 0, 1)
mat_file = './data/data/desc_validation.npz'
np.savez_compressed(mat_file, content=sw_comp_dt)
# %%
n_samples = 5
rnd_idx = np.random.choice(range(sw_comp_dt.shape[1]), n_samples, replace=False)
samples_dt = (np.take(sw_comp_dt, rnd_idx, 1)+1)/2
for i in range(n_samples):
    pyplot.subplot(4, n_samples, 1 + i)
    pyplot.axis('off')
    pyplot.imshow(samples_dt[0][i])
for i in range(n_samples):
    pyplot.subplot(4, n_samples, 1 + n_samples + i)
    pyplot.axis('off')
    pyplot.imshow(samples_dt[1][i])
for i in range(n_samples):
    pyplot.subplot(4, n_samples, 1 + 2*n_samples + i)
    pyplot.axis('off')
    pyplot.imshow(samples_dt[2][i])
pyplot.show()
# %%
