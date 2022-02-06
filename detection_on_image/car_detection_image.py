import cv2
from random import randrange
import os
# Change the current directory 
# to specified directory
cwd = os.getcwd() + "/detection_on_image/"
os.chdir(cwd)

#load img
#img_file = "car_image.jpg"
img_file = "cars_image.jpg"
img = cv2.imread(img_file)
#pre-trained classifier
classifier_file = 'car_detector.xml'
trained_car_data = cv2.CascadeClassifier(classifier_file)

#gray-scaling
gray_scaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#detect faces
car_coordinates = trained_car_data.detectMultiScale(gray_scaled_img)

#draw rectangles
for (x, y , w, h) in car_coordinates:
    cv2.rectangle(img, (x,y), (x+w, y+h), (randrange(256), randrange(256), randrange(256)), 2)
#show
cv2.imshow('car_pedestrian detector', img)
cv2.waitKey()