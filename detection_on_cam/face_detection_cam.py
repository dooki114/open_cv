#imports
import cv2
from random import randrange
import os
# Change the current directory 
# to specified directory
cwd = os.getcwd() + "/detection_on_cam/"
os.chdir(cwd)

#get a bunch of face data
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#webcam = cv2.VideoCapture("video file name') #video file
webcam = cv2.VideoCapture(0) #webcam

#iteration
while True:
    successful_frame_read, frame = webcam.read()
    #convert color to grayscale
    gray_scaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_coordinates = trained_face_data.detectMultiScale(gray_scaled_img)
    for (x, y , w, h) in face_coordinates:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 256, 0), 2)
    
    cv2.imshow("Clever Programmer Face Dectector", frame)
    key = cv2.waitKey(1) #default: wait on that frame until the key pressed / number: wait for that milliseconds per frame

    if key==81 or key==113: #Q or q key pressed
        break
        webcam.release()

#end
print("code completed")
