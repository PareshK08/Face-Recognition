import cv2 as cv

img =cv.imread(r'{Enter path of Image}')
img = cv.resize(img,(500,500))
#cv.imshow('Group',img)

gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
#cv.imshow('Gray',gray)

haar_cascade = cv.CascadeClassifier('haar_face.xml')
face_rect = haar_cascade.detectMultiScale(gray,
                scaleFactor=1.4,minNeighbors=4)

print(f'No. of faces : {len(face_rect)}')

for (x,y,w,h) in face_rect:
    cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),thickness=2)
    

cv.imshow('Detected Face',img)



cv.waitKey(0)
