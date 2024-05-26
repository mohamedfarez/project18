# project18
Face detection( CV2)


Detects faces in an image using the Haar Cascade algorithm and draws rectangles around the detected faces.

This script loads an image, converts it to grayscale, and then uses the Haar Cascade classifier to detect faces in the image. It then draws rectangles around the detected faces and displays the resulting image.

The script uses the following steps:
1. Load the Haar Cascade classifier for frontal face detection.
2. Load the image to be processed.
3. Convert the image to grayscale.
4. Detect faces in the grayscale image using the Haar Cascade classifier.
5. Draw rectangles around the detected faces on the original image.
6. Display the resulting image.

7. Detects and displays faces in a live video feed from the default camera.

This script uses the OpenCV library to capture video from the default camera, detect faces in each frame using a pre-trained Haar Cascade classifier, and draw rectangles around the detected faces on the video feed.

The script runs in an infinite loop, continuously capturing frames, detecting faces, and displaying the resulting video. The loop can be terminated by pressing the 'q' key.
