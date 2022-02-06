import cv2
from random import randrange
import os
# Change the current directory 
# to specified directory
cwd = os.getcwd() + "/detection_on_cam/"
os.chdir(cwd)

#load img
#img_file = "car_image.jpg"
#cam_file = cv2.VideoCapture("cars.mp4")
cam_file = cv2.VideoCapture("road_cam.mp4")

#pre-trained classifier
classifier_file = 'car_detector.xml'
trained_car_data = cv2.CascadeClassifier(classifier_file)

while True:
    successful_read, frame = cam_file.read()
    #convert color to grayscale
    gray_scaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    car_coordinates = trained_car_data.detectMultiScale(gray_scaled_img)
    for (x, y , w, h) in car_coordinates:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 256, 0), 2)
    
    cv2.imshow("AI Car Dectector", frame)
    key = cv2.waitKey(1) #default: wait on that frame until the key pressed / number: wait for that milliseconds per frame

    if key==81 or key==113: #Q or q key pressed
        break

#program ends
print("code completed!")