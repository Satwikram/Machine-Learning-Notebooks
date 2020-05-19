# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 19:34:30 2020

@author: SATWIK RAM K
"""
#Importing the Libraries
import numpy as np
import pandas as pd
import pickle


#Importing Dataset
dataset = pd.read_csv('Admission_Prediction.csv')

#Checking for NaN values
dataset.isna().sum()

#Mode of GRE Score fillna
dataset['GRE Score'].fillna(dataset['GRE Score'].mode()[0], inplace = True)

#Mode of Toefel Score fillna
dataset['TOEFL Score'].fillna(dataset['TOEFL Score'].mode()[0], inplace = True)

#Mode of FRating Score fillna
dataset['University Rating'].fillna(dataset['University Rating'].mean(), inplace = True)

#Taking X and Y
x = dataset.drop(['Serial No.', 'Chance of Admit'], axis = 1)

y = dataset['Chance of Admit']

from sklearn.pipeline import Pipeline


#To be used only if u want to Scale the Data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()


#Splitting the data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33, random_state = 100)

x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

#Linear Regression
from sklearn.linear_model import LinearRegression, Ridge, SGDRegressor
reg = LinearRegression()
reg1 = Ridge(alpha = 1, solver = "cholesky")
reg2 =  SGDRegressor( loss='squared_loss', penalty = None)


reg.fit(x_train, y_train)
reg1.fit(x_train, y_train)
reg2.fit(x_train, y_train.ravel())

#Predicting the output
y_pred = reg.predict(x_test)
y_pred1 = reg1.predict(x_test)
y_pred2 = reg2.predict(x_test)

#Calculating the accuracy of the model
from sklearn.metrics import r2_score
score = r2_score(y_pred, y_test)
score1 = r2_score(y_pred1, y_test)
score2 = r2_score(y_pred2, y_test)



