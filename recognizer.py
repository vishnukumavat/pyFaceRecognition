import cv2
import numpy as np

faceDetect = cv2.CascadeClassifier('C:\project 1\haarcascades\haarcascade_frontalface_default.xml')
# cam = cv2.VideoCapture('https://192.168.43.1:8080/video')
cam = cv2.VideoCapture(0)
rec = cv2.face.LBPHFaceRecognizer_create()
rec.read('C:\\project 1\\recognizer\\trainningData.yml')
identity = 0
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray, 1.3, 5)
    for(x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), [0, 255, 0, 255], 1)
        identity, conf = rec.predict(gray[y:y+h, x:x+w])
        print(conf, end="")
        print(identity)
        if conf > 70:
            cv2.putText(img, 'Unknown', (x, y + h), font, 2, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            if identity == 1:
                identity = "vishnu"
            if identity == 2:
                identity = "Rakesh"
            cv2.putText(img, str(identity), (x, y+h), font, 2, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, 'ComputerVision : ', (1, 20), cv2.FONT_HERSHEY_DUPLEX, 0.7, (129, 30, 43), 1, cv2.LINE_AA)
        cv2.putText(img, 'trial_model', (1, 40), cv2.FONT_HERSHEY_DUPLEX, 0.7, (129, 30, 43), 1, cv2.LINE_AA)

    cv2.imshow("Recognizer by VISHNU", img)
    if cv2.waitKey(1) == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
