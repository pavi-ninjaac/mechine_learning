# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 14:31:58 2020

@author: ninjaac
"""
from keras.datasets import mnist
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

(x_train,y_train), (x_test,y_test)=mnist.load_data()
some_digit=x_train[0]
#plt.imshow(some_digit,cmap = plt.cm.binary)
#plt.show()
y_train_5=(y_train==5)
print(np.shape(x_train))
x_train=x_train.reshape((60000,784))
print(np.shape(x_train))
# select a classifier and fit the model
from sklearn.linear_model import SGDClassifier
sgd=SGDClassifier(random_state=42)
sgd.fit(x_train,y_train_5)

#model performance mesure
from sklearn.model_selection import cross_val_score
score=cross_val_score(sgd,x_train,y_train_5,cv=5,scoring='accuracy')
print(score)

# more better prediction will be found by presicion and recall score
from sklearn.model_selection import cross_val_predict
y_train_pred = cross_val_predict(sgd, x_train, y_train_5, cv=3)
from sklearn.metrics  import precision_score,recall_score,f1_score
presicion=precision_score(y_train_5,y_train_pred)
print(f"precision {presicion}")
recall_sc=recall_score(y_train_5,y_train_pred)
print(f"Recall value {recall_sc}")
f1=f1_score(y_train_5,y_train_pred)
print(f" fi score value {f1}")
#presicion recall curve
y_scores = cross_val_predict(sgd, X_train, y_train_5, cv=3,
                             method="decision_function")
from sklearn.metrics import precision_recall_curve
precisions,recalls,thresholds=precision_recall_curve(y_train_5,y_scores)
def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.legend()
    plt.show()

plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
