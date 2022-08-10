# YAWN DETECTION - MODEL PREDICTION
import cv2
import numpy as np
from PIL import Image
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from mtcnn import MTCNN


loaded_model = load_model("Project_Saved_Models/Yawn_Detect_99acc.h5")
detector = MTCNN()
min_conf = 0.9

width1 = 100
height1 = 100

def start(path):
    data = []
    image = cv2.imread(path)
    imagee = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    detections = detector.detect_faces(imagee)
    for det in detections:
        if det['confidence'] >= min_conf:
            x, y, width, height = det['box']
            keypoints = det['keypoints']
            crop=imagee[y+height//2:y+height,x:x+width]
            
       
            resize_image=cv2.resize(crop,(height1,width1))
            cv2.imshow("crop",resize_image)
            cv2.waitKey(0)
            
            data.append(np.array(resize_image))
    x_test = np.array(data)/255
    print(x_test.shape)

    my_pred = loaded_model.predict(x_test)
    print(my_pred)

    my_pred = np.argmax(my_pred, axis=1)
    print(my_pred)
    my_pred = my_pred[0]
    print(my_pred)

    print("OUTPUT:")
    if my_pred == 0:
        print("RESULT: No Yawn")
    if my_pred == 1:
        print("RESULT: Yawn Detected")
    

if __name__=='__main__':
    from tkinter.filedialog import askopenfilename
    path=askopenfilename()
    start(path)


