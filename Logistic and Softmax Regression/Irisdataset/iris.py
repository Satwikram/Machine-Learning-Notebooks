# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 18:25:53 2019

@author: SATWIKRAM.K

"""
#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing datasets
dataset = pd.read_csv('Iris.csv')
replace_nums = {"Species": {"Iris-setosa":1}}
replace_nums1 = {"Species": {"Iris-versicolor": 2}}
replace_nums2 = {"Species": {"Iris-virginica": 3}}

dataset.replace(replace_nums, inplace = True)
dataset.replace(replace_nums1, inplace = True)
dataset.replace(replace_nums2, inplace = True)

x = dataset.iloc[:,1:5].values
y = dataset.iloc[:,5:].values

"""
#labeling the data
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
le = LabelEncoder()
y[0,] = le.fit_transform(y[0,])
oh = OneHotEncoder(categorical_features = [0])
y = oh.fit_transform(y).toarray()"""

#splitting the data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = 0.20, random_state = 0)

#importing the model
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(random_state = 0, criterion = 'entropy')
classifier.fit(x_train,y_train)

#predicting results
y_prediction = classifier.predict([[5.6, 3, 4.1, 1.3]])


import keras





















