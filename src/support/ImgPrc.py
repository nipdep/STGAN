import cv2
import pathlib 
import os


def img_resize(img, shape):
    return cv2.resize(img, shape, interpolation=cv2.INTER_LANCZOS4)

def load_and_save(in_path, out_path, shape):
    n = 0
    for file in os.listdir(in_path):
        n+=1
        img = cv2.imread(os.path.join(in_path, file))
        r_img = img_resize(img, shape)
        cv2.imwrite(os.path.join(out_path, "{0}.jpg".format(n)), r_img)

# load_and_save('E:/Data Science/Projects/intelli style transfer/STGAN/data/data/MSO_udni','E:/Data Science/Projects/intelli style transfer/STGAN/data/data/MSO_udnie', (128, 128))

