# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 19:18:09 2021

@author: kavinda
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Training the Simple Linear Regression model on the Training set
from sklearn.linear_model import LinearRegression
#create instace from class
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test and train set results
y_test_pred = regressor.predict(X_test)
y_train_pred = regressor.predict(X_train)

# Visualising the Training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, y_train_pred, color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualising the Test set results
plt.scatter(X_test, y_test, color = 'red')
#plt.plot(X_test, y_test_pred, color = 'blue')
# Both give the same result, same equation
plt.plot(X_train, y_train_pred, color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#predict 12 years of experiance
print(regressor.predict([[12]]))

#Getting the final linear regression equation with the values of the coefficients
print(regressor.coef_)
print(regressor.intercept_)
#Therefore, the equation of our simple linear regression model is:

#Salary = (9345.94 * YearsExperience) + 26816.19$$
