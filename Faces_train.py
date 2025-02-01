import os
import cv2 as cv
import numpy as np 


# try to gather images with same dimentions (Ex : 800 X 800 ) .
# Face in training image should be vesible clearly .  

haar_cascade = cv.CascadeClassifier('haar_face.xml')

People = []
DIR = r'{Enter Path of Training image folder}'
for i in os.listdir(DIR):
    People.append(i)

features =[]
labels = []

def create_train():
    for person in People:  
        path = os.path.join(DIR,person)
        label = People.index(person)
        for img in os.listdir(path):
            img_path = os.path.join(path,img)
            
            img_array = cv.imread(img_path)
            img_array=cv.resize(img_array,(500,500))
            gray = cv.cvtColor(img_array,cv.COLOR_BGR2GRAY)
            
            face_rect = haar_cascade.detectMultiScale(gray,scaleFactor=1.4,minNeighbors=4)
            
            for (x,y,w,h) in face_rect :
                faces_roi = gray[y:y+h,x:x+w]
                features.append(faces_roi)
                labels.append(label)
                
                
create_train()

print("Training Done ------------------------------")   


features = np.array(features,dtype='object')
labels = np.array(labels)

face_recoginizer = cv.face.LBPHFaceRecognizer_create()  

# Train the recognizer on the features list and the labels list

face_recoginizer.train(features,labels)

face_recoginizer.save('face_trained.yml')
np.save('features.npy',features) 
np.save('labels.npy',labels)        
                
                
            
