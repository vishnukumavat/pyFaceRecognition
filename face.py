import cv2
import numpy as np

faceDetect = cv2.CascadeClassifier('C:\project 1\haarcascades\haarcascade_frontalface_alt.xml')
eyeDetect = cv2.CascadeClassifier('C:\project 1\haarcascades\haarcascade_eye.xml')
cam = cv2.VideoCapture(0)

while True:
    ret, img = cam.read()   
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray, 1.3, 5)
    eyes = eyeDetect.detectMultiScale(gray, 1.3, 5)
    for(x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (164, 33, 25), 2)
    for (x, y, w, h) in eyes:
        cv2.rectangle(img, (x, y), (x + w, y + h), (164, 33, 25), 2)

    cv2.imshow("Face", img)
    if cv2.waitKey(1) == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
