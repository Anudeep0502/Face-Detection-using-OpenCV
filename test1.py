import os
import cv2

img = cv2.imread('img_1.jpg')

face_classifier = cv2.CascadeClassifier('haracascade_frontalface_alt.xml')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_classifier.detectMultiScale(gray, 1.1,4 )
print(faces)
if faces is ():
    print("No faces found")
for (x,y,w,h) in faces:
    cv2.rectangle(img, (x,y), (x+w,y+h))
    cv2.imshow('Face Detection', img)
    cv2.waitKey(0)
