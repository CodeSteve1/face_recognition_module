import pickle

import cv2
import os

import face_recognition
print("Starting facial recog")

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
folder="images"
PathList = os.listdir(folder)
imglst=[]
#na,e inside folder
for path in PathList:
    imglst.append(cv2.imread(os.path.join(folder,path)))

#encode file opening
file=open('encodefile.p','rb')
encodeListknownwithids = pickle.load(file)
encodeListKnown,id = encodeListknownwithids
file.close()
print("Encode file loaded...")


set = 0
while(True and set==0):
    success, img = cap.read()
    imgs = cv2.resize(img,(0,0),None,1,1)
    imgs = cv2.cvtColor(imgs,cv2.COLOR_BGR2RGB)

    facecurrFrame = face_recognition.face_locations(imgs)
    encodecurrframe = face_recognition.face_encodings(imgs,facecurrFrame)

    for encodeface,faceloc in zip(encodecurrframe,facecurrFrame):
        matches=face_recognition.compare_faces(encodeListKnown,encodeface)
        facedist = face_recognition.face_distance(encodeListKnown,encodeface)
        print(matches)
        print(facedist)
    cv2.imshow('face', img)
    cv2.waitKey(1)
