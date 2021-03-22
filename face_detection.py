import cv2 as cv
import numpy as np
import dlib
import io
import openface

#This is the model which detects the face
face_detector = dlib.get_frontal_face_detector()

#model which is used to find the 68 landamarks in the face
predictor_model = "shape_predictor_68_face_landmarks.dat"
face_pose_predictor = dlib.shape_predictor(predictor_model)

#face aligner model
face_aligner = openface.AlignDlib(predictor_model)

#Capturing video from the webcam
# capture = cv.VideoCapture(0)
#
# while True:
#     #reading single frame at a time
#     isTrue, frame = capture.read()
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
#     cv.imshow("Detected faces", frame)
#     if cv.waitKey(20) & 0xFF==ord('d'):
#         break
# capture.release()
# cv.destroyAllWindows()
################################################################################
image = cv.imread("Pictures/robert.jpg")

gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

#It gives the coordinates of the face in the image
faces = face_detector(gray, 1)

#drawing a rectangle in the faces
for i, face in enumerate(faces):
    predicted_landmarks = face_pose_predictor(gray, face)
    # for n in range(0, 68):
    #     x = predicted_landmarks.part(n).x
    #     y = predicted_landmarks.part(n).y
    #     cv.circle(image, (x, y), 1, (0, 255, 255), 1)
    alignedFace = face_aligner.align(534, image, face_rect, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
cv.imshow("Detected faces", image)
cv.imshow("Aligned face", alignedFace)
cv.waitKey(0)

################################################################################
