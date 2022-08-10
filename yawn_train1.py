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
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout,MaxPooling2D, BatchNormalization, GlobalMaxPooling2D, Concatenate

from tensorflow.keras.callbacks import ModelCheckpoint  # save the model

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import MaxPooling2D


from imutils.video import VideoStream
#identify the face
from imutils import face_utils
from threading import Thread
import numpy as np
import pandas as pd

import imutils
import time
# import dlib
import cv2
from mtcnn import MTCNN
import cv2

from tensorflow.keras.applications import InceptionV3 as Net
from tensorflow.keras import optimizers
from tensorflow.keras import Model
from tensorflow.keras.applications import VGG16


path_train = "yawn_dataset/"
my_folder = 2
height1=100
width1=100

detector = MTCNN()
min_conf = 0.9


def my_function(path_train):
    images = []
    labels = []
    for i in range(my_folder):                # iterate each folder(in DATASET folder)
        path = path_train + '{0}/'.format(i)    # selecting folder
        list_of_images = os.listdir(path)        # listing images in the folder

        for a in list_of_images:                 # iterate each image on the list of images
            print(path+a)
            image = cv2.imread(path+a)
            
            imagee = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            detections = detector.detect_faces(imagee)
            for det in detections:
            # draw rectangle on detected face in image
                if det['confidence'] >= min_conf:
                    x, y, width, height = det['box']
                    keypoints = det['keypoints']
                    crop=imagee[y+height//2:y+height,x:x+width]
                    # kernel = np.array([[0, -1, 0],
                    # [-1, 5,-1],
                    # [0, -1, 0]])
                    # crop = cv2.filter2D(src=crop, ddepth=-1, kernel=kernel)
                    # cv2.imshow('out',crop)
                    # cv2.waitKey(1)
                    resize_image=cv2.resize(crop,(height1,width1))
                    #print("resize_image :",resize_image.shape)
                    
                    images.append(np.array(resize_image))
                    labels.append(i)

    return images, labels


images, labels = my_function(path_train)
print(len(images))
print(len(labels))


# #creating dataframe 
df=pd.DataFrame(columns=['images','labels'])
df['images']=images
df['labels']=labels

# print(df.head())

#df save to pickle file
df.to_pickle("yawn_data.pkl")
####################################################

####################################################LOADING SAVED PICKLE FILE
# #read pickle file as dataframe
df2 = pd.read_pickle('yawn_data.pkl')
# print(df2)


train,test=train_test_split(df2,test_size=0.2,random_state=42,shuffle=True)#adding train_size
print("Train shape(df) :",train.shape)
print("Test shape(df) :",test.shape)


print("type(train['images'] :",type(train['images']))  ####pandas.core.series.Series (type)
print("train['images'].shape :",train['images'].shape) 
print("train['labels'].shape :",train['labels'].shape)

    

x_train=np.stack(train['images'].values)/255
y_train=np.stack(train['labels'].values)
print("x_train shape :",x_train.shape) 
print("y_train shape :",y_train.shape) 
print("type of  x_train",type(x_train)) #numpy array
print("type of  y_train",type(y_train))



x_test=np.stack(test['images'].values)/255
y_test=np.stack(test['labels'].values)

y_train = to_categorical(y_train, 2)
y_test=to_categorical(y_test,2)

print("after categorical")
print(y_train.shape)             

def model1():
    input_shape=(100,100,3)

    conv_base = Net(weights="imagenet", include_top=False, input_shape=input_shape)
   
    # dropout_rate = 0.2
    model = Sequential()
    model.add(conv_base)
    model.add(GlobalMaxPooling2D(name="gap"))
    # model.add(layers.Flatten(name="flatten"))
    # if dropout_rate > 0:
    #     model.add(Dropout(dropout_rate, name="dropout_out"))
    conv_base.trainable = False
    model.add(Dense(256, activation="relu"))
    model.add(Dense(2, activation="softmax"))
    model.compile(loss="categorical_crossentropy",optimizer='adam',metrics=["acc"])
    print(model.summary())
    return model

def model2():
    #step1 Initializing CNN
    classifier = Sequential()

    # step2 adding 1st Convolution layer and Pooling layer
    classifier.add(Convolution2D(32,(3,3),input_shape = (100,100,3), activation = 'relu'))

    classifier.add(MaxPooling2D(pool_size = (2, 2)))

    # step3 adding 2nd convolution layer and polling layer
    classifier.add(Convolution2D(32,(3,3), activation = 'relu'))

    classifier.add(MaxPooling2D(pool_size = (2, 2)))


    #step4 Flattening the layers
    classifier.add(Flatten())

    #step5 Full_Connection

    classifier.add(Dense(units=32,activation = 'relu'))

    classifier.add(Dense(units=64,activation = 'relu'))

    classifier.add(Dense(units=128,activation = 'relu'))

    classifier.add(Dense(units=256,activation = 'relu'))

    classifier.add(Dense(units=256,activation = 'relu'))

    classifier.add(Dense(units=2,activation = 'softmax'))

    #step6 Compiling CNN
    classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    print(classifier.summary())
    return classifier

   
def model3():
    vgg = VGG16(input_shape=[100,100] + [3], weights='imagenet', include_top=False)

    # don't train existing weights
    for layer in vgg.layers:
        layer.trainable = False

    x = Flatten()(vgg.output)
    prediction = Dense(2, activation='softmax')(x)
    model = Model(inputs=vgg.input, outputs=prediction)

    #printing model summary
    print(model.summary())

    #compiling the model
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model

# model=model1()    
# model=model2()
model=model3()

# Training
# saving the model
checkpoint = ModelCheckpoint(
    "Project_Saved_Models/Yawn_Detect.h5", monitor="val_accuracy", save_best_only=True, verbose=1)

# training
epochs = 30
history = model.fit(x_train, y_train, batch_size=32,validation_data=(x_test,y_test),
                    epochs=epochs, callbacks=[checkpoint])


