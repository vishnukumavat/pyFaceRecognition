import cv2
from pip._vendor.distlib.compat import raw_input
import os
import numpy as np
from PIL import Image


faceDetect = cv2.CascadeClassifier('C:\project 1\haarcascades\haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0)
# cam = cv2.VideoCapture('https://192.168.43.1:8080/video')

identity = raw_input('enter user id : ')

sampleNum = 0
while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray, 1.3, 5)
    for(x, y, w, h) in faces:
        sampleNum = sampleNum+1
        cv2.imwrite("dataSet/User."+str(identity)+"."+str(sampleNum)+".jpg", gray[y:y+h, x:x+w])
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.waitKey(100)

    cv2.imshow("Face", img)
    cv2.waitKey(1)
    if sampleNum > 50:
        break

cam.release()

recognizer = cv2.face.LBPHFaceRecognizer_create()
path = 'dataSet'


def getImagesWithID(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    facetrainer = []
    IDs = []
    for imagePath in imagePaths:
        faceImg = Image.open(imagePath).convert('L')
        faceNP = np.array(faceImg, 'uint8')
        ID = int(os.path.split(imagePath)[-1].split('.')[1])
        facetrainer.append(faceNP)
        IDs.append(ID)
        cv2.imshow("training", faceNP)
        cv2.waitKey(20)

    return IDs, facetrainer


IDs, facetrainer = getImagesWithID(path)
recognizer.train(facetrainer, np.array(IDs))
recognizer.save('recognizer/trainningData.yml')
cv2.destroyAllWindows()
