import pandas as pd
import numpy as np
import cv2 as cv
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier

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

#Training a model
sample_face = X_train.iloc[4]
knn_cls = KNeighborsClassifier()
knn_cls.fit(X_train, y_train)

#Predicting the sample face
prediction = knn_cls.predict([sample_face])
result = False
if prediction == y_train[4]:
    result = True
print(f"Our model predicted {prediction} which is {result}")
