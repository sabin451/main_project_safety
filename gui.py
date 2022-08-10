from tkinter import *
from tkinter import messagebox
from PIL import ImageTk, Image
from tkinter.filedialog import askopenfilename
import cv2
import numpy as np
import math
import os
import time
#from predict import main1
import threading
from imutils.video import VideoStream
import datetime
from datetime import datetime
from threading import Thread
from imutils import face_utils  # image processing
from scipy.spatial import distance as dist  # scientific_mathematical prblms
from imutils.video import VideoStream
import imutils
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import paho.mqtt.client as mqtt
from datetime import date
from mtcnn import MTCNN
import playsound

a = Tk()
a.title("Driver Drowsiness Detection")
a.geometry("800x550")
a.maxsize(800, 550)
a.minsize(800,550)


eye_closure_model = load_model("Project_Saved_Models/Eye_Detect_94acc.h5")
yawn_detection_model=load_model("Project_Saved_Models/Yawn_Detect_99acc.h5")
detector = MTCNN()
min_conf = 0.9

width1 = 100
height1 = 100

def sound_alarm(path):
    print("--------")
    playsound.playsound(path)

def detect_eye_closure(frame0):
    global count

    data = []
    imagee = cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB)
    detections = detector.detect_faces(imagee)
    for det in detections:
        if det['confidence'] >= min_conf:
            x, y, width, height = det['box']
            keypoints = det['keypoints']
            crop=imagee[y:y+height//2,x:x+width]

            # cv2.imshow("crop",crop)
            # cv2.waitKey(1)
            resize_image=cv2.resize(crop,(height1,width1))
            
            data.append(np.array(resize_image))
    x_test = np.array(data)/255
    #print(x_test.shape)

    my_pred = eye_closure_model.predict(x_test)
    #print(my_pred)

    my_pred = np.argmax(my_pred, axis=1)
    #print(my_pred)
    my_pred = my_pred[0]
    # print(my_pred)

    #print("OUTPUT:")
    if my_pred == 0:
        print("RESULT: Eyes Closed")
        count+=1
    if my_pred == 1:
        print("RESULT: Eyes Open")

    return count


def yawn_detection(frame0):
    global counter

    data1 = []
    image = cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB)
    detections = detector.detect_faces(image)
    for det in detections:
        if det['confidence'] >= min_conf:
            x, y, width, height = det['box']
            keypoints = det['keypoints']
            crop=image[y+height//2:y+height,x:x+width]
            # cv2.imshow("crop",crop)
            # cv2.waitKey(1)
       
            resize_image=cv2.resize(crop,(height1,width1))
            
            data1.append(np.array(resize_image))
    x_test = np.array(data1)/255
    #print(x_test.shape)

    my_pred = yawn_detection_model.predict(x_test)
    #print(my_pred)

    my_pred = np.argmax(my_pred, axis=1)
    #print(my_pred)
    my_pred = my_pred[0]
    # print(my_pred)

    #print("OUTPUT:")
    if my_pred == 0:
        print("RESULT: No Yawn")
    if my_pred == 1:
        print("RESULT: Yawn Detected")
        counter+=1

    return counter


def main():
    global count,counter,max_time,start_time,s_flag
    s_flag=0
    count = 0  # eye
    counter = 0  # yawn
    max_time = 1 * 20
    start_time = time.time()
    eye_thresh=3  #5
    yawn_thresh=3

    cap0 = cv2.VideoCapture(0)

    while(cap0.isOpened()):
        if s_flag==1:
            break

        ret, frame0 = cap0.read()

        frame0=cv2.resize(frame0,(500,500),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)

        val1=detect_eye_closure(frame0)
        val2=yawn_detection(frame0)

        delta = time.time()-start_time
        #print(delta)

        if delta >= max_time:
            print("-------------------------------")
            if val1>eye_thresh and val2>yawn_thresh:
                print("[DANGER!!!] : Drowsiness Detected")
                cv2.putText(frame0, "  ONE MINUTE --Eye Closure Count: {}".format(val1), (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame0, "  ONE MINUTE --Yawning Count: {}".format(val2), (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame0, "[DANGER!!!] : Drowsiness Detected", (10, 440),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                sound_alarm("alarm.wav")
                #global start_time
                start_time = time.time()
                count=0
                counter=0
            else:
                print("[Safe] : Enjoy Your Driving")
                cv2.putText(frame0, "  ONE MINUTE --Eye Closure Count: {}".format(val1), (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame0, "  ONE MINUTE --Yawning Count: {}".format(val2), (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                start_time = time.time()
                count=0
                counter=0
                
            print("--------------------------------")
        else:
            print("Eye Blink Count : %d" %val1)
            print("Yawning Count : %d" %val2)


        cv2.imshow("Output",frame0)
        cv2.waitKey(1)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap0.release()
    cv2.destroyAllWindows()




def stop():
    global s_flag
    s_flag=1



def Check():
    global f
    f.pack_forget()

    # # Add image file
    # # Create Canvas
    # canvas1 = Canvas( a, width = 400,
    #                  height = 400)
    # canvas1.pack(fill = "both", expand = True)
    # bg=ImageTk.PhotoImage(Image.open("pic.jpg"))
      
    # # Display image
    # canvas1.create_image( 20, 20, image = bg, 
    #                      anchor = "nw")
      
    # # Create Buttons
    # button1 = Button( a, text = "Exit")

    # start_button = Button(f1, text="Capture Video",height=3,width=11 ,command=lambda: threading.Thread(target=main).start(), bg="aquamarine")
    # start_button.pack(anchor=CENTER,pady=200)
    # end_button = Button(f1, text="Stop",height=1,width=11 ,command=lambda: threading.Thread(target=stop).start(), bg="OrangeRed2")
    # end_button.pack()

    # # Display Buttons
    # button1_canvas = canvas1.create_window( 100, 10, 
    #                                        anchor = "nw",
    #                                        window = button1)
      

    f = Frame(a, bg="white")
    f.pack(side="top", fill="both", expand=True)

    global f1
    f1 = Frame(f, bg="lavender")
    f1.place(x=0, y=0, width=800, height=550)
    f1.config()

    front_image = Image.open("pic.jpg")
    front_photo = ImageTk.PhotoImage(front_image.resize((a.winfo_width(), a.winfo_height()), Image.ANTIALIAS))
    front_label = Label(f1, image=front_photo,bg="white")
    front_label.image = front_photo
    front_label.pack()


    start_button = Button(f1, text="Capture Video",height=3,width=11 ,command=lambda: threading.Thread(target=main).start(), bg="aquamarine")
    start_button.place(x=300, y=190)
    #pack(anchor=CENTER,pady=200)
    end_button = Button(f1, text="Stop",height=1,width=11 ,command=lambda: threading.Thread(target=stop).start(), bg="OrangeRed2")
    end_button.place(x=300, y=450)#pack()
  


  

def Home():
    global f
    f.pack_forget()

    f = Frame(a, bg="light sky blue")
    f.pack(side="top", fill="both", expand=True)

    front_image = Image.open("Project_Extra/p3.jpg")
    front_photo = ImageTk.PhotoImage(front_image.resize((a.winfo_width(), a.winfo_height()), Image.ANTIALIAS))
    front_label = Label(f, image=front_photo)
    front_label.image = front_photo
    front_label.pack()

    home_label = Label(f, text="Driver Drowsiness Detection",
                       font="arial 30", bg="white")
    home_label.place(x=140, y=230)


f = Frame(a, bg="light sky blue")
f.pack(side="top", fill="both", expand=True)

front_image1 = Image.open("Project_Extra/p3.jpg")
front_photo1 = ImageTk.PhotoImage(front_image1.resize((800, 550), Image.ANTIALIAS))
front_label1 = Label(f, image=front_photo1)
front_label1.image = front_photo1
front_label1.pack()

home_label = Label(f, text="Driver Drowsiness Detection",
                   font="arial 30", bg="white")
home_label.place(x=140, y=230)

m = Menu(a)
m.add_command(label="Home", command=Home)
checkmenu = Menu(m)
m.add_command(label="Check", command=Check)
a.config(menu=m)


a.mainloop()
















