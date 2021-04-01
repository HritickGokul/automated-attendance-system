import pandas as pd
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
################################################################################
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_validate
from sklearn.metrics import roc_auc_score
from sklearn import preprocessing
from sklearn.metrics import f1_score
################################################################################
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
################################################################################
import dlib
import io
from imutils.face_utils import FaceAligner
import face_recognition
import os
################################################################################
face_detector = dlib.get_frontal_face_detector()
predictor_model = "shape_predictor_68_face_landmarks.dat"
face_pose_predictor = dlib.shape_predictor(predictor_model)

#face aligner model
face_aligner = FaceAligner(face_pose_predictor, desiredFaceWidth = 256)

def rescale_frame(frame, scale = 0.2):
    #images, videos, live videos
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)
    return cv.resize(frame, dimensions, interpolation = cv.INTER_AREA)

#Get the data
data = pd.read_csv('train.csv')
del data["Unnamed: 0"]

#Shuffle the data
for i in range(10):
    data = data.sample(frac = 1, random_state = 42).reset_index(drop = True)

#Setting index for the data
data = pd.DataFrame(data, index = [i for i in range(len(data))])

#Split the data into train and test set
strat_split = StratifiedShuffleSplit(n_splits = 5, test_size = 0.2, random_state = 42)
for train_index, test_index in strat_split.split(data,data["label"]):
    train_set = data.loc[train_index]
    test_set = data.loc[test_index]

#Split the labels separately from the features of the train set
X_train = train_set.drop(["label"], axis = 1)
X_train.index = [i for i in range(len(X_train))]
y_train = train_set["label"]
y_train.index = [i for i in range(len(y_train))]
y_train = preprocessing.label_binarize(y_train, classes = ['Chandrakala', 'Hrithick Gokul', 'Sai Prasad', 'Sakunthala', 'Sanjana'])

#Training a model
scaler = StandardScaler()
sample_face = X_train.iloc[4]
models = {"KNearestNeighbors":KNeighborsClassifier(),
          "SupportVectorClassifier":SVC(kernel = 'rbf', gamma = 0.5, C = 0.1, random_state = 42),
          "StochasticGradientDescent":SGDClassifier(random_state = 42),
          "RandomForestClassifier":RandomForestClassifier(random_state = 42),
          "VotingClassifier":VotingClassifier(estimators=[('KNearestNeighbors', KNeighborsClassifier()), ('StochasticGradientDescent', SGDClassifier(random_state = 42))], voting='soft')}
for name, model in models.items():
    print("------Training------")
    try:
        classifier = model
        X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
        classifier.fit(X_train_scaled, y_train)
        y_predd = cross_val_predict(classifier, X_train, y_train, cv = 3)
        print(f"Train f1 score using {name} classifier is {f1_score(y_train, y_predd, average = 'macro')}.")
        print(f"Train Cross val score using {name} classifer is {cross_val_score(classifier, X_train_scaled, y_train, cv = 3)}.")
    except ValueError:
        y_train = train_set["label"]
        y_train.index = [i for i in range(len(y_train))]
        classifier.fit(X_train_scaled, y_train)
        y_predd = cross_val_predict(classifier, X_train, y_train, cv = 3)
        print(f"Train f1 score using {name} classifier is {f1_score(y_train, y_predd, average = 'macro')}.")
        print(f"Train Cross val score using {name} classifer is {cross_val_score(classifier, X_train_scaled, y_train, cv = 3)}.")

    #Predicting the test set using the trained model
    # X_test = test_set.drop(["label"], axis = 1)
    # X_test.index = [i for i in range(len(X_test))]
    # y_test = test_set["label"]
    # y_test.index = [i for i in range(len(y_test))]
    # y_test = preprocessing.label_binarize(y_test, classes = ['Chandrakala', 'Hrithick Gokul', 'Sai Prasad', 'Sakunthala', 'Sanjana'])
    #
    # try:
    #     if name is "VotingClassifier":
    #         print(f"Test score of {name} is {classifier.score(X_test, y_test)}")
    #     else:
    #         #predicted values for the test set
    #         y_train_pred = cross_val_predict(classifier, X_test, y_test, cv=3)
    #         print(f"Test f1 score using {name} classifier is {f1_score(y_test, y_train_pred, average = 'macro')}")
    # except ValueError or NotImplementedError:
    #     y_test = test_set["label"]
    #     y_test.index = [i for i in range(len(y_test))]
    #     if name is "VotingClassifier":
    #         print(f"Test score of {name} is {classifier.score(X_test, y_test)}")
    #     else:
    #         #predicted values for the test set
    #         y_train_pred = cross_val_predict(classifier, X_test, y_test, cv=3)
    #         print(f"Test f1 score using {name} classifier is {f1_score(y_test, y_train_pred, average = 'macro')}")
    img = cv.imread("Pictures/sai2.jpeg")
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
        face_enc = list(face_recognition.face_encodings(alignedFace)[0])
        print(face_enc)
        pred = classifier.predict([face_enc])
        print(pred)
        # cv.putText(img, str(pred), (100,100), cv.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), thickness = 2 )
        # cv.imshow(f"{name}'s Prediction", img)
        # cv.waitKey(0)
