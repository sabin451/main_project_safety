# import essential libraries
from imutils.video import VideoStream
import time
import datetime
from datetime import datetime
import math
from threading import Thread
import cv2
import numpy as np  # high level mathematical operations
import dlib  # machine learning(face detection)
from imutils import face_utils  # image processing
from scipy.spatial import distance as dist  # scientific_mathematical prblms
from imutils.video import VideoStream
import imutils
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import paho.mqtt.client as mqtt
import threading
from datetime import datetime
from datetime import date
from mtcnn import MTCNN

#client= mqtt.Client()


eye_closure_model = load_model("Project_Saved_Models/Eye_Detect_94acc.h5")
yawn_detection_model=load_model("Project_Saved_Models/Yawn_Detect_99acc.h5")
detector = MTCNN()
min_conf = 0.9

width1 = 100
height1 = 100

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
    #print(my_pred)

    #print("OUTPUT:")
    if my_pred == 0:
        print("RESULT: No Yawn")
    if my_pred == 1:
        print("RESULT: Yawn Detected")
        counter+=1

    return counter


if __name__=='__main__':
    global count,counter,max_time,start_time
    count = 0  # eye
    counter = 0  # yawn
    max_time = 1 * 60
    start_time = time.time()
    eye_thresh=5
    yawn_thresh=4

    cap0 = cv2.VideoCapture(0)

    while(cap0.isOpened()):

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





















# def yawn_detection(frame):
#     # print("I AM HERE____YAWN DETECTION")

#     global counter
#     global client

#     lip_LAR = 0.2
#     lip_per_frame = 4

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     # face detection in grayscale image
#     faces = detector(gray)
#     for (i, face) in enumerate(faces):
#         # facial feature coordinates of dlib
#         lips = [60, 61, 62, 63, 64, 65, 66, 67]  # indexes of lip
#         # grayscale image & detected face
#         # output is facial landmark coordinates
#         point = predictor(gray, face)
#         # convert facial landmark coordinates to numpy array
#         points = face_utils.shape_to_np(point)
#         lip_point = points[lips]
#         LAR = calculate_lip(lip_point)  # lip aspect ratio

#         # tight fitting convex boundary around the lip points
#         lip_hull = cv2.convexHull(lip_point)
#         cv2.drawContours(frame, [lip_hull], -1,
#                          (0, 255, 0), 1)  # shape draw

#         global start_time
#         delta = time.time()-start_time
#         print(delta)
#         if delta < max_time:
#             if LAR > lip_LAR:
#                 counter += 1
#                 print(counter)
#                 if counter > lip_per_frame:
#                     cv2.putText(frame, "YAWNING DETECTED!", (10, 440),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
#                     global yawning_counter
#                     yawning_counter += 1
#             else:
#                 counter = 0

#             cv2.putText(frame, "LAR: {:.2f}".format(LAR), (500, 30),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
#         else:
#             cv2.putText(frame, "  ONE MINUTE --Yawning Count: {}".format(yawning_counter), (10, 60),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
#             # time.sleep(8)
#             if yawning_counter > 5:
#                 cv2.putText(frame, " Drowsiness Detected", (200, 400),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
#                 # time.sleep(4)
#                 my_value1 = "drowsy"
#                 client.connect("broker.hivemq.com", 1883, 60)
#                 client.publish("drowsy", my_value1)
                

#             yawning_counter = 0
#             start_time = time.time()
#         cv2.imshow("Frame", frame)


# def eye_blink(frame):
#     global count
#     global client
#     width = 100
#     height = 100

#     data = []
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     equalized = cv2.equalizeHist(gray)
#     pic_face = face_cascade.detectMultiScale(
#         equalized, scaleFactor=1.1, minNeighbors=4)
#     for (x, y, w, h) in pic_face:
#         cv2.rectangle(equalized, (x, y), (x+w, y+h), (0, 0, 255), 2)
#         # cropping the face portion & save it to a folder
#         crop_face = equalized[y:y + h, x:x + w]
#     image_from_array = Image.fromarray(crop_face, 'L')
#     resize_image = image_from_array.resize((height, width))
#     data.append(np.array(resize_image))
#     x_test = np.array(data)
#     x_test = tf.expand_dims(x_test, axis=-1)

#     my_pred = loaded_model.predict(x_test)
#     my_pred = np.argmax(my_pred, axis=1)
#     my_pred = my_pred[0]
#     print(my_pred)

#     global start_time
#     delta = time.time()-start_time
#     print(delta)
#     if delta < max_time:
#         # print("OUTPUT:")
#         if my_pred == 0:

#             count += 1
#             cv2.putText(frame, " Eyes Closed--Blink Count: {}".format(count), (10, 30),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
#             print("RESULT: Eyes Closed")
#         if my_pred == 1:
#             cv2.putText(frame, " Eyes Open", (10, 30),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
#             print("RESULT: Eyes Open")
#     else:
#         cv2.putText(frame, "  ONE MINUTE --Blink Count: {}".format(count), (10, 30),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
#         # time.sleep(3)
#         if count > 20:
#             # time.sleep(8)
#             cv2.putText(frame, " Drowsiness Detected", (200, 400),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
#             # time.sleep(4) 
#             my_value2 = "drowsy"
#             client.connect("broker.hivemq.com", 1883, 60)
#             client.publish("drowsy", my_value2)  
#         count = 0
#         start_time = time.time()

#     cv2.imshow('Frame', frame)







    # # vs = cv2.VideoCapture(0)
    # vs = VideoStream(src=0).start()
    # count = 0  # eye
    # counter = 0  # yawn
    # yawning_counter = 0
    # max_time = 1 * 60
    # start_time = time.time()
    # while True:
    #     frame = vs.read()
    #     frame = imutils.resize(frame, width=100, height=100)
    #     print(frame.shape)

    #     # yawn_detection(frame)
    #     # eye_blink(frame)

    #     # show the frame
    #     # cv2.imshow("Frame", frame)
    #     key = cv2.waitKey(1) & 0xFF

    #     # if the `q` key was pressed, break from the loop
    #     if key == ord("q"):
    #         break

    # cv2.destroyAllWindows()
    # vs.stop()


##############  mqtt start ############





##############  mqtt end ############
