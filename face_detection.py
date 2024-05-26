"""
Detects faces in an image using the Haar Cascade algorithm and draws rectangles around the detected faces.

This script loads an image, converts it to grayscale, and then uses the Haar Cascade classifier to detect faces in the image. It then draws rectangles around the detected faces and displays the resulting image.

The script uses the following steps:
1. Load the Haar Cascade classifier for frontal face detection.
2. Load the image to be processed.
3. Convert the image to grayscale.
4. Detect faces in the grayscale image using the Haar Cascade classifier.
5. Draw rectangles around the detected faces on the original image.
6. Display the resulting image.
"""
import cv2

#   use  Haar Cascade 
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# import image
img = cv2.imread('photo/image.jpg')  

#  make photo become gray
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# face detection
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# draw square in face detection
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

# show the image
cv2.imshow('Faces', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
