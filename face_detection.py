import cv2
from random import randrange
trained_face = cv2.CascadeClassifier('haarcascade.xml')

# img = cv2.imread('download.jpeg')
webcam = cv2.VideoCapture(0)

while True:
    successful_frame_read, frame = webcam.read()
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_coordinates = trained_face.detectMultiScale(grayscaled_img)
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (randrange(256),
                                                  randrange(256), randrange(256)), 3)

    cv2.imshow('Python Face Detection App', frame)

    key = cv2.waitKey(1)

    if key == 81 or key == 113:
        break


webcam.release()

print("Code executed successfully")
