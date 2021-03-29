import pandas as pd
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn import preprocessing
from sklearn.multiclass import OneVsOneClassifier
from sklearn.metrics import f1_score


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
train_set["indexx"] = [i for i in range(len(train_set))]
train_set.set_index("indexx")

#Split the labels separately from the features of the train set
X_train = train_set.drop(["label"], axis = 1)
y_train = train_set["label"]
y_train.index = [i for i in range(len(y_train))]
y_train = preprocessing.label_binarize(y_train, classes = ['Chandrakala', 'Hrithick Gokul', 'Sai Prasad', 'Sakunthala', 'Sanjana'])

#Training a model
scaler = StandardScaler()
sample_face = X_train.iloc[4]
knn_cls = KNeighborsClassifier()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
knn_cls.fit(X_train_scaled, y_train)

print(cross_val_score(knn_cls, X_train_scaled, y_train, cv = 3))


#Predicting the sample face
test_set["indexx"] = [i for i in range(len(test_set))]
test_set.set_index("indexx")

#Split the labels separately from the features of the train set
X_test = test_set.drop(["label"], axis = 1)
y_test = test_set["label"]
y_test.index = [i for i in range(len(y_test))]
y_test = preprocessing.label_binarize(y_test, classes = ['Chandrakala', 'Hrithick Gokul', 'Sai Prasad', 'Sakunthala', 'Sanjana'])

#predicted values for the test set
y_train_knn_pred = cross_val_predict(knn_cls, X_test, y_test, cv=3)
print(y_train_knn_pred)
print(f1_score(y_test, y_train_knn_pred, average="macro"))
