# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 21:46:20 2021

@author: kavinda
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Data.csv')

#take all raws in all cols except the last col
x = dataset.iloc[:, :-1].values
#task all raws of last col
y = dataset.iloc[:, -1].values

# Taking care of missing data
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
#ignore col index 0, get mean of all other cols
imputer.fit(x[:, 1:])
#replace MT values with mean
x[:, 1:] = imputer.transform(x[:, 1:])

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
#replace first col with 3 bin cols
x = np.array(ct.fit_transform(x))

from sklearn.preprocessing import LabelEncoder
#replace yes and no with 1 and 0
le = LabelEncoder()
y = le.fit_transform(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1)

#Normalization and Standalization
#Normalization good if values have normal distribution
#Standalization good for any distribution
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train[:, 3:] = sc.fit_transform(x_train[:, 3:])
#use same scaller form train dataset, only transform for test data test
x_test[:, 3:] = sc.transform(x_test[:, 3:])