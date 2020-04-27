# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#confision matrix
from sklearn.metrics import confusion_matrix

# Importing the dataset
dataset = pd.read_csv(r'C:\Users\ninjaac\Desktop\P14-Machine-Learning-AZ-Template-Folder\Part 3 - Classification\Section 15 - K-Nearest Neighbors (K-NN)\K_Nearest_Neighbors\Social_Network_Ads.csv')
X = dataset.iloc[:,[2,3]].values
y = dataset.iloc[:, 4].values


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import  SGDClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier

k_cla=KNeighborsClassifier(n_neighbors=5)
l_cla=SGDClassifier()
svm_clas=SVC()
D_cla=DecisionTreeClassifier()
# tring the voting classifier
vote_cla=VotingClassifier(
    estimators=[('kc',k_cla),('lc',l_cla),('sc',svm_clas),('dc',D_cla)],
                            voting='hard')
    
vote_cla.fit(X_train,y_train)
y_vote_pred=vote_cla.predict(X_test)
    
from sklearn.metrics import accuracy_score

for classi in (k_cla,l_cla,svm_clas,D_cla):
    classi.fit(X_train,y_train)
    y_pred=classi.predict(X_test)
    
    print(classi.__class__.__name__,accuracy_score(y_test,y_pred))
"""
KNeighborsClassifier 0.83
SGDClassifier 0.32
SVC 0.75
DecisionTreeClassifier 0.9
"""

    
print('voting classifier accuracy score',accuracy_score(y_test,y_vote_pred ))    
# voting classifier accuracy score 0.91

# tring voting classifier with soft voting
svm_clas_soft=SVC(probability=True)  
vote_cla_soft=VotingClassifier(
    estimators=[('kc',k_cla),('sc',svm_clas_soft),('dc',D_cla)],
                            voting='soft')
    
vote_cla_soft.fit(X_train,y_train)
y_vote_pred_soft=vote_cla_soft.predict(X_test)
    
print('voting classifier accuracy  soft score',accuracy_score(y_test,y_vote_pred_soft ))    
#voting classifier accuracy  soft score 0.9

from sklearn.ensemble import RandomForestClassifier

ren_class=RandomForestClassifier(n_estimators=100,criterion='gini',max_depth=4)
ren_class.fit(X_train,y_train)
y_pred_ren=ren_class.predict(X_test)
print('random forest classifier accuracy score',accuracy_score(y_test,y_vote_pred_soft ))    
#random forest classifier accuracy score 0.9
    
    
    
