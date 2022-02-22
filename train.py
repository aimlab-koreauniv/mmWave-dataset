import os
import tensorflow as tf
from glob import glob
import numpy as np
import random
import tqdm
import cv2
import pandas as pd
from models import makemodel
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
import argparse
import models
from tensorflow.keras.applications.vgg19 import VGG19

# =======================================================
# [add parser]
# =======================================================
parser = argparse.ArgumentParser(description='Train setting')

parser.add_argument('--model', type=str, required=True, help='Set train model', choices=['vgg', 'resnet', 'efficientnet'])
parser.add_argument('--epoch', type=int, required=True, help='Set train epoch')

args = parser.parse_args()
print(args.model)

# =======================================================
#Set input image size
# =======================================================
IMAGE_SIZE = 224

# =======================================================
#Load CNN model
# =======================================================
train_model = args.model
if train_model=='vgg':
    model = makemodel.Vgg19()
elif train_model=='resnet':
    model = makemodel.Resnet()
else:
    model = makemodel.Efficientnet()

# =======================================================
#Load mmWave radar dataset
# =======================================================
def read_image(filepath):
    return cv2.imread(os.path.join(data_dir, filepath)) 

def resize_image(image, image_size):
    return cv2.resize(image.copy(), image_size, interpolation = cv2.INTER_AREA)

gesture_types = ['ap', 'aw', 'lr', 'rl']
data_dir = os.path.join(os.getcwd(), 'data')
train_dir = os.path.join(data_dir)
train_data = []

for gesture_id, sp in enumerate(gesture_types):
    for file in os.listdir(os.path.join(train_dir, sp)):
        train_data.append(['{}/{}'.format(sp, file), gesture_id, sp])
        
train = pd.DataFrame(train_data, columns=['File', 'ID','Type'])

X = np.zeros((train.shape[0], IMAGE_SIZE, IMAGE_SIZE, 3))

for i, file in enumerate(train['File'].values):
    image = read_image(file)
    if image is not None:
        X[i] = resize_image(image, (IMAGE_SIZE, IMAGE_SIZE))
X /= 255.
y = train['ID'].values

# =======================================================
#Split dataset into a train dataset and a test dataset
# =======================================================
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state = 1)

# =======================================================
#Set train model 
# =======================================================
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# =======================================================
#Learning dataset
# =======================================================
hist = model.fit(x_train, y_train, batch_size=32, epochs=args.epoch)

# =======================================================
#Save result and model (accuracy, loss per epoch)
# =======================================================
result = pd.DataFrame(hist.history)
result.to_csv("./train_hist.csv")
model.save('./mmWave.h5', save_format='h5')