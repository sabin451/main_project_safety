##############EYE CLOSURE DETECTION - MODEL CREATION
import cv2  # image processing
from PIL import Image  # image processing
import os  # operating system interaction
import numpy as np  # mathematical operations
from sklearn.model_selection import train_test_split  # split dataset
# convert label numbers to proper vector
from tensorflow.keras.utils import to_categorical
import tensorflow as tf  # deep learning

from tensorflow.keras.models import Sequential  # model creation
# layers of sequential model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout

from tensorflow.keras.layers import Convolution2D, Dense, Input, Flatten, Dropout, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D, Concatenate

from tensorflow.keras.callbacks import ModelCheckpoint  # save the model

from imutils.video import VideoStream
#identify the face
from imutils import face_utils
from threading import Thread
import numpy as np

import imutils
import time
# import dlib
import cv2
from mtcnn import MTCNN
import cv2

#from keras.applications import InceptionV3 as Net



path_train = "eye_dataset/"
my_folder = 2
height1 =100
width1 = 100

detector = MTCNN()
min_conf = 0.9


def my_function(path_train):
    images = []
    labels = []
    for i in range(my_folder):                # iterate each folder(in DATASET folder)
        path = path_train + '{0}/'.format(i)    # selecting folder
        list_of_images = os.listdir(path)        # listing images in the folder

        for a in list_of_images:#[:100]:                 # iterate each image on the list of images
            print(path+a)
            image = cv2.imread(path+a)
            
            imagee = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            detections = detector.detect_faces(imagee)
            for det in detections:
            # draw rectangle on detected face in image
                if det['confidence'] >= min_conf:
                    x, y, width, height = det['box']
                    keypoints = det['keypoints']
                    crop=imagee[y:y+height//2,x:x+width]
                    # crop=cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
                    # kernel = np.array([[0, -1, 0],
                    # [-1, 5,-1],
                    # [0, -1, 0]])
                    # crop = cv2.filter2D(src=crop, ddepth=-1, kernel=kernel)
                    # cv2.imshow('out',crop)
                    # cv2.waitKey(1)
                    resize_image=cv2.resize(crop,(height1,width1))
                    # kernel = np.array([[0, -1, 0],
                    # [-1, 5,-1],
                    # [0, -1, 0]])
                    # resize_image = cv2.filter2D(src=resize_image, ddepth=-1, kernel=kernel)
                    
                    images.append(np.array(resize_image))
                    labels.append(i)
    return np.array(images), np.array(labels)


images, labels = my_function(path_train)
print(images.shape)
print(labels.shape)

# Train-Test Splitting

x_train, x_test, y_train, y_test = train_test_split(
    images, labels, test_size=0.2)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# print("after dimension change")
# print(x_train.shape)
# print(x_test.shape)
x_test=np.stack(x_test)
x_train=np.stack(x_train)

x_train = np.reshape(x_train,(x_train.shape[0],height1,width1,3))/255
x_test = np.reshape(x_test,(x_test.shape[0],height1,width1,3))/255

print(x_train.shape)
print(x_test.shape)


# """If your training data uses classes as numbers,
# to_categorical will transform those numbers in proper vectors"""
y_train = to_categorical(y_train, 2)
y_test = to_categorical(y_test, 2)
print("after categorical")
print(y_train.shape)
print(y_test.shape)
from tensorflow.keras import applications
from tensorflow.keras import optimizers, losses, activations, models

from CNN import model1
    
model=model1(2)

# Training
# saving the model
checkpoint = ModelCheckpoint(
    "Eye_Detect1.h5", monitor="val_accuracy", save_best_only=True, verbose=1)

# training
epochs = 100
history = model.fit(x_train, y_train, batch_size=32,validation_data=(x_test,y_test),
                    epochs=epochs, callbacks=[checkpoint])


