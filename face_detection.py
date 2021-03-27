import cv2 as cv
import numpy as np
import pandas as pd
import dlib
import io
from imutils.face_utils import FaceAligner
import face_recognition
import os

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
feature_labels = []
for i in range(128):
    feature_labels.append("m" + str(i))
feature_labels.append("label")
print("----labels created----")

df = pd.DataFrame(columns = feature_labels)
labels = []

people = ['Chandrakala', 'Hrithick Gokul', 'Sai Prasad', 'Sakunthala', 'Sanjana']

DIR = r'C:\Users\sanjana\Desktop\Automated Attendace System\Photos'

def create_train():
    for person in people:
        path = os.path.join(DIR, person)
        label = people.index(person)

        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            img = cv.imread(img_path)
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            faces = face_detector(gray, 1)
            for i, face in enumerate(faces):
                #Finding the landmarks
                predicted_landmarks = face_pose_predictor(gray, face)
                # for n in range(0, 68):
                #     x = predicted_landmarks.part(n).x
                #     y = predicted_landmarks.part(n).y
                #     cv.circle(image, (x, y), 1, (0, 255, 255), 1)

                #Aligning the face
                alignedFace = face_aligner.align(img, gray, face)

                #Face embedding
                try:
                    face_enc = list(face_recognition.face_encodings(alignedFace)[0])
                    face_enc.append(people[label])
                    df.loc[len(df.index)] = face_enc
                    print(f"----person {label}----")
                except IndexError:
                    continue

    df.to_csv('train.csv')
    return df
result_table = create_train()
print(result_table)
cv.waitKey(0)

################################################################################
