import os
import matplotlib.pyplot as plt
import cv2
import glob
import random

# Get image data from directory
img1 = glob.glob("data/ap/*")
img2 = glob.glob("data/aw/*")
img3 = glob.glob("data/lr/*")
img4 = glob.glob("data/rl/*")

# Read image data and convert into RBG format
def read_img(file_path):
    img_arr = cv2.imread(file_path)
    return cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)

# Set random number
img_arrs = []
img_num = range(1,1000)

# 4 random images out of 1000 images for each gesture
for i in random.sample(img_num,4):
    img_arrs.append(read_img(img1[i]))
    img_arrs.append(read_img(img2[i]))
    img_arrs.append(read_img(img3[i]))
    img_arrs.append(read_img(img4[i]))

# Set plot
rows = 4; cols = 4
fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(cols*4,rows*4))

for num in range(1,rows*cols+1):
    fig.add_subplot(rows, cols, num)
    idx = num-1
    plt.imshow(img_arrs[idx], aspect='auto')
    plt.axis("off")
fig.tight_layout()    

# Labeling
gesture = ['Gesture 1','Gesture 2','Gesture 3','Gesture 4']

for folder_idx, ax in enumerate(axes[0]):
    ax.set_title(gesture[folder_idx])

for idx, ax in enumerate(axes.flat):
    ax.set_xticks([])
    ax.set_yticks([])
plt.show()