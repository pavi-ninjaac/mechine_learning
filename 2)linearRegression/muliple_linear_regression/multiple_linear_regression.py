# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
'''
author:pavi
date: 10.4.2020 22:44
Language: python
Field: mechine learning
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#for preprocessing
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
#for model
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv(r'C:\Users\ninjaac\Desktop\P14-Machine-Learning-AZ-Template-Folder\Part 2 - Regression\Section 5 - Multiple Linear Regression\Multiple_Linear_Regression\50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

#data preproceeing
#to use onehotencoding we must convert the string values to interger using
#lableencoder and then use the 
#onehotencoder otherwise it throudhs the error  'cant convert string to float' 
lable=LabelEncoder()
X[:,3]=lable.fit_transform(X[:,3])

one=OneHotEncoder(categorical_features=[3])
X=one.fit_transform(X).toarray()

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 2/5, random_state = 0)

#create the object and fit the model
li=LinearRegression()
li.fit(X_train,y_train)

#predic the data

y_pred=li.predict(X_test)
#due to multiple independant variaple we cant visualize it now
'''
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
'''
