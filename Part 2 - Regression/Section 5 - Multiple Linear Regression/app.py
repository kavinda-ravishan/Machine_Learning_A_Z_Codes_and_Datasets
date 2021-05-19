# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 19:20:36 2021

@author: kavinda
"""

#Multiple liner regression : y = b0 + (b1 * x1) + (b2 * x2) + (b3 * x3) + ...

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Encoding categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
# apply onehot encoing to column 3 and pass rest column to result
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Training the Multiple Linear Regression model on the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
#make array vertial (reshape)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

#https://www.superdatascience.com/pages/ml-regression-bonus-2
new_val = np.array([[0, 1, 0, 66051.52, 182645.56, 118148.2]])
#get single prediction
print(regressor.predict(new_val))

print(regressor.coef_)
print(regressor.intercept_)

#Therefore, the equation of our multiple linear regression model is:
#Profit = 86.6 * Dummy_State_1 - 873 * Dummy_State_2 + * Dummy_State_3 + 0.773 * R&D_Spend + 0.0329 * Administration + 0.0366 * Marketing_Spend + 42467.53$$

