from collections import defaultdict
from scipy.stats import itemfreq
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from skimage import feature
from PIL import Image as IMG
import numpy as np
import pandas as pd 
import operator
import cv2
import os 

nrows = None
train = False

if train:
    images_path = '/media/deeplearning/E8DC7CDFDC7CAA08/Bac/avito/data/competition_files/train_jpg/'
    df = pd.read_csv("/home/deeplearning/Kaggle/avito/input/train.csv", usecols=["image"], nrows=nrows)
else:
    images_path = '/media/deeplearning/E8DC7CDFDC7CAA08/Bac/avito/test_jpg/data/competition_files/test_jpg/'
    df = pd.read_csv("/home/deeplearning/Kaggle/avito/input/test.csv", usecols=["image"], nrows=nrows)


def getSize(filename):
    try:
        filename = images_path + str(filename) + ".jpg"
        st = os.stat(filename)
        return st.st_size
    except:
        return 0

def getDimensions(filename):
    try:
        filename = images_path + str(filename) + ".jpg"
        img_size = IMG.open(filename).size
        return img_size 
    except:
        return (0,0)

df['image_size'] = df['image'].apply(getSize)
df['temp_size'] = df['image'].apply(getDimensions)
df['width'] = df['temp_size'].apply(lambda x : x[0])
df['height'] = df['temp_size'].apply(lambda x : x[1])
df = df.drop(['temp_size'], axis=1)

if train:
    df.to_csv("./extracted_features/train_img_meta.csv", index=False)
else:
    df.to_csv("./extracted_features/test_img_meta.csv", index=False)
