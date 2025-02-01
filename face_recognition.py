import numpy as np
import cv2 as cv

haar_cascade = cv.CascadeClassifier('haar_face.xml')

People = ['Abhay', 'Arigit Singh','Paresh', 'RDJ', 'Sachin Tendulkar']

#features = np.load('features.npy')
#labels = np.load('labels.npy')

face_recoginizer = cv.face.LBPHFaceRecognizer_create() 

face_recoginizer.read('face_trained.yml')

img = cv.imread(r'D:\Img\Test\Sachin.jpg')
img = cv.resize(img,(500,500))
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
#cv.imshow('Person',gray)
# Detect the face in image 
face_rect = haar_cascade.detectMultiScale(gray,scaleFactor=1.4,minNeighbors=4)

for (x,y,w,h) in face_rect:
    face_roi = gray[y:y+h,x:x+h]
    
    label , comfidence = face_recoginizer.predict(face_roi)
    print(f'Label = {People[label]} with a confidence {comfidence}')
    
    cv.putText(img,str(People[label]),(x,y),cv.FONT_HERSHEY_COMPLEX , 1.0 ,(0,255,0),thickness=2 )
    cv.rectangle(img, (x,y),(x+w,y+h),(0,255,0),thickness=2)

cv.imshow("Detected Image",img)
cv.waitKey(0)    
    


