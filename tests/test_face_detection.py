"""
Tests for the face detection functionality.

This module contains unit tests for the face detection functionality implemented in the `face_detection` module.
"""
import unittest
import cv2
from face_detection import face_cascade, detect_faces

class TestFaceDetection(unittest.TestCase):

    def test_face_cascade_loaded(self):
        self.assertIsInstance(face_cascade, cv2.CascadeClassifier)

    def test_detect_faces_valid_image(self):
        img = cv2.imread('tests/test_image.jpg')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detect_faces(gray)
        self.assertIsInstance(faces, list)
        self.assertGreaterEqual(len(faces), 0)

    def test_detect_faces_no_faces(self):
        img = cv2.imread('tests/no_face_image.jpg')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detect_faces(gray)
        self.assertEqual(len(faces), 0)

    def test_detect_faces_invalid_input(self):
        with self.assertRaises(TypeError):
            detect_faces(123)

if __name__ == '__main__':
    unittest.main()
