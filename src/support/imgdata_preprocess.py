def process_img(img):
  img = tf.image.decode_jpeg(img, channels=3) 
  img = tf.image.convert_image_dtype(img, tf.float32) 
  return tf.image.resize(img, config.IMAGE_SIZE)

def process_path(file_path):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    #print(fp)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, size=(128, 128))
    return img

def train_gen():
    lower, higher, root_path, n = 1, 2923, './data/data/StyleDataset', 2900
    idx = np.array(range(lower, min(higher, lower+n)))
    for i in idx:
        #i = random.randint(lower, higher)
        random_num = random.randint(lower, higher)
        random_bool = random.randint(0,1)
        if random_bool:
            if random_num == int(i):
                random_num = random.randint(lower, higher)
        else:
            random_num = max(random.randint(1,10), int(i)-5)
        img1_det = stenc_df.loc[i, ['path', 'style_code']]
        img2_det = stenc_df.loc[random_num, ['path', 'style_code']]

        label = 0
        if img1_det['style_code'] == img2_det['style_code']:
            label = 1
        #print(os.path.join(root_path, img1_det['path']), os.path.join(root_path, img2_det['path']))
        try :
            img1 = process_path(os.path.join(root_path, img1_det['path']))
            img2 = process_path(os.path.join(root_path, img2_det['path']))
            yield img1, img2, label
        except:
            print(f"Error in file {img1_det['path']}")
            continue

def val_gen():
    lower, higher, root_path, n = 2923, 3164, './data/data/StyleDataset', 200
    # idx = np.random.choice(range(lower, higher), n, replace=False, seed=111)
    # for i in idx:
    idx = np.array(range(lower, min(higher, lower+n)))
    for i in idx:
        #i = random.randint(lower, higher)
        random_num = random.randint(lower, higher)
        random_bool = random.randint(0,1)
        if random_bool:
            if random_num == int(i):
                random_num = random.randint(lower, higher)
        else:
            random_num = max(random.randint(1,10), int(i)-5)
        img1_det = stenc_df.loc[i, ['path', 'style_code']]
        img2_det = stenc_df.loc[random_num, ['path', 'style_code']]

        label = 0
        if img1_det['style_code'] == img2_det['style_code']:
            label = 1
        #print(os.path.join(root_path, img1_det['path']), os.path.join(root_path, img2_det['path']))
        try :
            img1 = process_path(os.path.join(root_path, img1_det['path']))
            img2 = process_path(os.path.join(root_path, img2_det['path']))
            yield img1, img2, label
        except:
            print(f"Error in file {img1_det['path']}")
            continue

# image resize and rescale pipeline
resize_and_rescale = tf.keras.Sequential([
    prep.Resizing(config.IMG_HEIGHT, config.IMG_WIDTH),
    prep.Normalization()
])

# image augmentation pipeline
data_augmentation = tf.keras.Sequential([
    prep.RandomContrast(0.2),
    prep.RandomFlip("horizontal_and_vertical"),
    prep.RandomCrop(config.IMG_HEIGHT, config.IMG_WIDTH),
    prep.RandomRotation(0.3, fill_mode='nearest', interpolation='bilinear'),
    prep.RandomZoom(height_factor=(-0.2, 0.2), width_factor=(-0.2, 0.2), fill_mode='nearest', interpolation='bilinear')
])

# data_augmentation = tf.keras.Sequential([
#   prep.RandomFlip("horizontal_and_vertical"),
#   prep.RandomRotation(0.2),
# ])

def prepare(ds, shuffle=False, augment=False):
    # ds = ds.map(lambda x: tf.py_function(process_path, [x], [tf.float32, tf.float32, tf.int32]),
    #                         num_parallel_calls=tf.data.AUTOTUNE)

    # ds = ds.map(lambda x1, x2, y: (process_path(x1), process_path(x2), y),
    #             num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.map(lambda x1, x2, y: (resize_and_rescale(x1), resize_and_rescale(x2), y),
                num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.cache()
    
    if shuffle:
        ds = ds.shuffle(1000)

    ds = ds.batch(16)

    if augment:
        ds = ds.map(lambda x1, x2, y: (data_augmentation(x1, training=True), data_augmentation(x2, training=True), y), 
                    num_parallel_calls=tf.data.AUTOTUNE)
    
    return ds.prefetch(buffer_size=tf.data.AUTOTUNE)
