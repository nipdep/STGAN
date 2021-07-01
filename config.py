# all the model training and architecture parameters

#general directories
LOG_DIR = './logs'
GAN_LOG_DIR = './logs/gan'
DESC_ROOT_DIR = './data/data/Desc_dataset/'  
DESC_ENC_DIR = './data/data/Desc_dataset/style_enc.csv'
GAN_DATASET_DIR = './data/data/comp_matrics.npz'
GAN_DATASET_URL = 'https://anonfiles.com/vah4ba3au4/comp_matrics_npz'

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
DESC_ALPHA = 0.2
DESCS_LATENT_SIZE = 64

# GAN model paramters
GAN_LATENT_SIZE = 100
GAN_BATCH_SIZE = 32
GAN_EPOCHS = 100
GAN_ALPHA = 0.05
GAN_BETA = 0.7
GAN_BP = 300

