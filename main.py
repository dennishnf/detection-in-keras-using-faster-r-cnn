# -*- coding: utf-8 -*-

"""

Object Detection using Faster R-CNN

Caution: modify the path /content/drive/My Drive/detection-in-keras-using-faster-r-cnnn 
in the lines of code of this file, according to your path in Drive. Eg: if the location 
of your folder is in the main path of your Google Drive under the name faster-r-cnn-colab, 
then modify the last path to: /content/drive/My Drive/faster-r-cnn-colab

"""

#%%

# Downloading the dataset

#%%

# Download manually and unzip the daatset from:
# https://docs.google.com/uc?export=download&id=1TVx7005znK0QgK-prd5maH5BJmHpf2Wq

#%%

# Preparing Libraries

#%%

import os
os.environ['KERAS_BACKEND'] = 'tensorflow'

import tensorflow
import keras
tensorflow.test.gpu_device_name()


#%%

print(tensorflow.__version__)
print(keras.__version__)

# should print something like:
# 1.10.0
# 2.1.5

#%%

folder_location = "/home/dennis/Downloads/detection-in-keras-using-faster-r-cnn/"
dataset_location = "/home/dennis/Downloads/detection-in-keras-using-faster-r-cnn/dataset-blood-cells/"

#%%

# Train

#%%

# If you want to use with your own dataset, modify the train_images.txt file according your drive path. Eg:

# ...
# /home/dennis/Downloads/detection-in-keras-using-faster-r-cnn/dataset-blood-cells/train_images/BloodImage_00001.jpg,68,315,286,480,WBC
# /home/dennis/Downloads/detection-in-keras-using-faster-r-cnn/dataset-blood-cells/train_images/BloodImage_00001.jpg,346,361,446,454,RBC
# /home/dennis/Downloads/detection-in-keras-using-faster-r-cnn/dataset-blood-cells/train_images/BloodImage_00001.jpg,53,179,146,299,RBC
# /home/dennis/Downloads/detection-in-keras-using-faster-r-cnn/dataset-blood-cells/train_images/BloodImage_00001.jpg,449,400,536,480,RBC
# ...

#%%

import train_frcnn

from train_frcnn import *

#%%

training(train_path = dataset_location+"train_images.txt",
         input_weight_path = "",
         parser = "simple",
         num_rois = 32,
         network = "resnet50",
         horizontal_flips = False,
         vertical_flips = False,
         rot_90 = False,
         num_epochs = 2,
         config_filename = folder_location+"config.pickle",
         output_weight_path = folder_location+"model_frcnn.hdf5")

#%%

# Test

#%%

import test_frcnn

from test_frcnn import *

#%%

testing(test_path = dataset_location+"test_images",
        num_rois = 32,
        config_filename = folder_location+"config.pickle",
        network = "resnet50",
        weigths_file = folder_location+"model_frcnn.hdf5",
        output_folder = dataset_location+"test_images_result/")

#%%

# Visualization

#%%

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#%%

img=mpimg.imread(dataset_location+"test_images_result/0.png")
plt.imshow(img, cmap = 'gray')
plt.axis('off')
plt.show()

#%%

img=mpimg.imread(dataset_location+"test_images_result/1.png")
plt.imshow(img, cmap = 'gray')
plt.axis('off')
plt.show()

#%%

img=mpimg.imread(dataset_location+"test_images_result/2.png")
plt.imshow(img, cmap = 'gray')
plt.axis('off')
plt.show()

#%%