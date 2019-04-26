import numpy as np
import cv2
import pickle

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_eye.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('./recognizers/face-trainner.yml')

labels = {"person_name": 1}
with open('pickles/face-labels.pickle', 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}

cap = cv2.VideoCapture(0)


while(True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        id_, conf = recognizer.predict(roi_gray)
        font = cv2.FONT_HERSHEY_SIMPLEX
        name = labels[id_]
        color = (255, 255, 255)
        stroke = 2
        if conf < 100:
            #font = cv2.FONT_HERSHEY_SIMPLEX
            #name = labels[id_]
            #color = (255, 255, 255)
            #stroke = 2
            conf = " {0}%".format(round(100 - conf))
        else:
            name = 'unknown'
            conf = " {0}%".format(round(100 - conf))
            
        cv2.putText(frame,name, (x+5, y-5), font, 1, color, stroke)
        cv2.putText(frame, str(conf), (x+10, y+10), font, 1, color, 1)

        img_item = "7.png"
        cv2.imwrite(img_item, roi_gray)

        color = (255, 0, 0)
        stroke = 2
        end_cord_x = x + w
        end_cord_y = y + h
        cv2.rectangle(frame, (x,y), (end_cord_x, end_cord_y), color, stroke)

        cv2.imshow('frame', frame) 												# Hit 'q' on the keyboard to quit!
        if cv2.waitKey(20) & 0xFF == ord('q'):
                break


cap.release()
cv2.destroyAllWindows()
