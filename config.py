# all the model training and architecture parameters

#general directories
LOG_DIR = './logs'
DESC_ROOT_DIR = './data/data/Desc_dataset/'  
DESC_ENC_DIR = './data/data/Desc_dataset/style_enc.csv'


# image parameters
#from support.notebooks.data_pipe import IMG_HEIGHT


IMG_WIDTH, IMG_HEIGHT = 128, 128
IMAGE_SIZE = (128, 128)

#desctiminator data pipeline

DESC_TRAIN_SIZE = 8000
DESC_SHUFFLE_BUFFER = 1000
DESC_BATCH_SIZE = 16
DESC_EPOCHS = 10
DESC_INIT_LR = 0.02
M = 1

