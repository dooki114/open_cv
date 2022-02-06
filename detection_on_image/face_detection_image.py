#imports
import cv2
from random import randrange
import os

# Change the current directory 
# to specified directory
cwd = os.getcwd() + "/detection_on_image/"
os.chdir(cwd)

#get a bunch of face data
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#sample images
#img = cv2.imread('RDJ.jpeg')
img = cv2.imread('multiple_faces.jpg')

#convert the target image color to gray
gray_scaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#detect faces
face_coordinates = trained_face_data.detectMultiScale(gray_scaled_img)
#print(face_coordinates)

#draw rectangles
for (x, y , w, h) in face_coordinates:
    cv2.rectangle(img, (x,y), (x+w, y+h), (randrange(256), randrange(256), randrange(256)), 2)

#save/show img
'''
filename = "face_detected_output.jpg"
cv2.imwrite(filename, img)
'''
cv2.imshow('Face Detector', img)
cv2.waitKey()

#end
print("code completed")