import cv2 as cv
import numpy as np
import dlib
import io
from imutils.face_utils import FaceAligner
import face_recognition

#This is the model which detects the face
face_detector = dlib.get_frontal_face_detector()

#model which is used to find the 68 landamarks in the face
predictor_model = "shape_predictor_68_face_landmarks.dat"
face_pose_predictor = dlib.shape_predictor(predictor_model)

#face aligner model
face_aligner = FaceAligner(face_pose_predictor, desiredFaceWidth = 256)

################################################################################
#Capturing video from the webcam
# capture = cv.VideoCapture(0)
#
# while True:
#     #reading single frame at a time
#     isTrue, frame = capture.read()
#
#     # win = dlib.image_window()
#     # win.set_image(frame)
#
#     #Converting thr frame to gray image
#     gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
#
#     #It gives the coordinates of the face in the image
#     faces = face_detector(gray, 1)
#
#     #drawing a rectangle in the faces
#     for i, face in enumerate(faces):
#         predicted_landmarks = face_pose_predictor(gray, face)
#         for n in range(0, 68):
#             x = predicted_landmarks.part(n).x
#             y = predicted_landmarks.part(n).y
#             cv.circle(frame, (x, y), 1, (0, 255, 255), 1)
#         alignedFace = face_aligner.align(frame, gray, face)
#     cv.imshow("Landmarks in the face", frame)
#     if cv.waitKey(20) & 0xFF==ord('d'):
#         break
# capture.release()
# cv.destroyAllWindows()
################################################################################
image = cv.imread("Pictures/robert.jpg")
cv.imshow('Original Image', image)


# win = dlib.image_window()
# win.set_image(image)

#Converting the image to gray
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

#It gives the coordinates of the face in the image
faces = face_detector(gray, 1)

for i, face in enumerate(faces):
    #drawing a rectangle in the detected faces
    # win.add_overlay(face)

    #Finding the landmarks
    predicted_landmarks = face_pose_predictor(gray, face)
    # for n in range(0, 68):
    #     x = predicted_landmarks.part(n).x
    #     y = predicted_landmarks.part(n).y
    #     cv.circle(image, (x, y), 1, (0, 255, 255), 1)

    #Aligning the face
    alignedFace = face_aligner.align(image, gray, face)

    #Face embedding
    face_encodings = face_recognition.face_encodings(alignedFace)[0]
    print(len(face_encodings))

cv.imshow("Aligned face", alignedFace)
cv.waitKey(0)

################################################################################
