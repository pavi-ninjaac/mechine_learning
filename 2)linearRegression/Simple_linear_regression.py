# -*- coding: utf-8 -*-
'''
author:pavi
date: 10.4.2020 22:22
Language: python
Field: mechine learning
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#for model
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv(r'C:\Users\ninjaac\Desktop\P14-Machine-Learning-AZ-Template-Folder\Part 2 - Regression\Section 4 - Simple Linear Regression\Simple_Linear_Regression\Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

#create the object and fit the model
li=LinearRegression()
li.fit(X_train,y_train)

#predic the data

y_pred=li.predict(X_test)

#visualization using matplotlib for training data
plt.scatter(X_train,y_train,color='red',edgecolor='blue')
plt.plot(X_train,li.predict(X_train))
plt.title('simple linear regression')
plt.xlabel('Experience')
plt.ylabel('salary')
plt.show()

#visualization using matplotlib for test data
plt.scatter(X_test,y_test,color='red',edgecolor='blue')
plt.plot(X_train,li.predict(X_train))
plt.title('simple linear regression')
plt.xlabel('Experience')
plt.ylabel('salary')
plt.show()
