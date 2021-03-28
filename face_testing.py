import pandas as pd
import numpy as np
import cv2 as cv
from sklearn.model_selection import StratifiedShuffleSplit

data = pd.read_csv('train.csv')
del data["Unnamed: 0"]
for i in range(10):
    data = data.sample(frac = 1, random_state = 42).reset_index(drop = True)
strat_split = StratifiedShuffleSplit(n_splits = 5, test_size = 0.2, random_state = 42)
for train_index, test_index in strat_split.split(data,data["label"]):
    train_set = data.loc[train_index]
    test_set = data.loc[test_index]
