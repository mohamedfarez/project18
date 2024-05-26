"""
Detects and displays faces in a live video feed from the default camera.

This script uses the OpenCV library to capture video from the default camera, detect faces in each frame using a pre-trained Haar Cascade classifier, and draw rectangles around the detected faces on the video feed.

The script runs in an infinite loop, continuously capturing frames, detecting faces, and displaying the resulting video. The loop can be terminated by pressing the 'q' key.
"""
import cv2

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# open camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow('Faces', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
