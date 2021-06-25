# all the model training and architecture parameters

#general directories
LOG_DIR = './logs'
GAN_LOG_DIR = './logs/gan'
DESC_ROOT_DIR = './data/data/Desc_dataset/'  
DESC_ENC_DIR = './data/data/Desc_dataset/style_enc.csv'
GAN_DATASET_DIR = './data/data/comp_matrics.npz'

# image parameters
#from support.notebooks.data_pipe import IMG_HEIGHT


IMG_WIDTH, IMG_HEIGHT = 128, 128
IMAGE_SIZE = (128, 128)
IMAGE_SHAPE = (128, 128, 3)

#desctiminator data pipeline

DESC_TRAIN_SIZE = 8000
DESC_SHUFFLE_BUFFER = 1000
DESC_BATCH_SIZE = 16
DESC_EPOCHS = 10
DESC_INIT_LR = 0.02
M = 1

# GAN model paramters
GAN_ALPHA = 0.1
GAN_BETA = 0.5

