# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 22:53:34 2020

@author: ninjaac
"""

# imports for getting the dataset
import os
import tarfile
from six.moves import urllib

#imports for future usage
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# download the dataset as tar file
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/" 
HOUSING_PATH = r"E:\housing_data"
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"
def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH): 
     if not os.path.isdir(housing_path):        
         os.makedirs(housing_path)
     tgz_path=os.path.join(housing_path,"housing.tgz")
     urllib.request.urlretrieve(housing_url,tgz_path)
     housing_data=tarfile.open(tgz_path)
     housing_data.extractall(path=housing_path)
     housing_data.close()
    
fetch_housing_data()    

# load the dataset
def load_data(path=HOUSING_PATH):
    dataset_path=os.path.join(path,"housing.csv")
    return pd.read_csv(dataset_path)

# get the data
dataset=load_data()

# first function need to run after get the dataset to get the info about that
dataset.head()
dataset.info()

# inspect the catocarical values

dataset['ocean_proximity'].value_counts()
# summary of the data
dataset.describe()

# analyse by visuvalizing

dataset.hist(bins=20,figsize=(20,40))
plt.show()

# preparing the train and test set

from sklearn.model_selection import train_test_split
train,test=train_test_split(dataset,test_size=0.2,random_state=42)

# income is the very important data so evry range of data must be present in the test set
# so we create the cut in the data

dataset['income_catogiry']=pd.cut(dataset["median_income"],
                                   bins=[0., 1.5, 3.0, 4.5, 6., np.inf],                               
                                   labels=[1, 2, 3, 4, 5])
dataset["income_catogiry"].hist()                                           

from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42) 
for train_index, test_index in split.split(dataset, dataset["income_catogiry"]):    
    strat_train_set = dataset.loc[train_index]    
    strat_test_set = dataset.loc[test_index]

for set_ in (strat_train_set, strat_test_set):    
    set_.drop("income_catogiry", axis=1, inplace=True)

# find the correlation

# plot the geographical values
plt.scatter(x=dataset["longitude"],y=dataset["latitude"],color="red",alpha=0.1)

dataset.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)

plt.scatter(x=dataset["longitude"],y=dataset["latitude"],alpha=0.4,
            s=dataset['population']/100,c=dataset["median_house_value"],
            cmap=plt.get_cmap('jet'),label='population')
plt.legend()
plt.show()

# find the correlation
corrrelation_matrix=dataset.corr()
corrrelation_matrix["median_house_value"].sort_values(ascending=False)
#find the scatter_matrix

from pandas.plotting import scatter_matrix
attributes = ["median_house_value", "median_income", "total_rooms",              
              "housing_median_age"] 
scatter_matrix(dataset[attributes], figsize=(12, 8))

plt.scatter(x=dataset["median_income"],y=dataset["median_house_value"],color='blue',alpha=0.1)
"""
# calculate the per values
dataset["rooms_per_household"] = dataset["total_rooms"]/dataset["households"] 
dataset["bedrooms_per_room"] = dataset["total_bedrooms"]/dataset["total_rooms"]
dataset["population_per_household"]=dataset["population"]/dataset["households"]

correlation_matrix_2=dataset.corr()
correlation_matrix_2["median_house_value"].sort_values(ascending=False)

housing_labels = strat_train_set["median_house_value"].copy()
housing = strat_train_set.drop("median_house_value", axis=1) 

#data preprocesing data cleaning

from sklearn.preprocessing import Imputer
imputer=Imputer(strategy='median')

housing_ocen=dataset.drop('ocean_proximity',axis=1)
imputer.fit(housing_ocen)
imputer.statistics_
X=imputer.transform(housing_ocen)

housing_tr=pd.DataFrame(X,columns=housing_ocen.columns)

# handling the catogarical values
housng_cato=dataset[["ocean_proximity"]]

from sklearn.preprocessing import LabelEncoder
label=LabelEncoder()
housung_cato_encoded=label.fit_transform(housng_cato)

#if we habdle it with arrays first we need to transfer dats to labelencoded values before
# using onehot encoder

from sklearn.preprocessing import OneHotEncoder
one=OneHotEncoder()
housing_cato_1hot=one.fit_transform(housng_cato)

housing_cato_1hot.toarray()
"""
#instead of doing these steps anually we will make functions and pipe lines

from sklearn.base import BaseEstimator,TransformerMixin
rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

class AttributeAdder(BaseEstimator,TransformerMixin):
    def __init__(self,add_pedroom_per_room=True):
        self.add_pedroom_per_room=add_pedroom_per_room
    def fit(self,X):
        return self
    def transform(self,X):
        rooms_per_household =X[:,rooms_ix]/X[:,households_ix]
        papulation_per_household=X[:,population_ix]/X[:,households_ix]
        if self.add_pedroom_per_room==True:
            bedroom_per_room=X[:,bedrooms_ix]/X[:,rooms_ix]
            return np.c_[X,rooms_per_household,papulation_per_household,bedroom_per_room]
        else:
            return np.c_[X,rooms_per_household,papulation_per_household]
        
        
attr_adder = AttributeAdder(add_pedroom_per_room=False) 
housing_extra_attribs = attr_adder.transform(dataset.values)

# handling missing values
from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

def fullPipline(housing_data):
    housing_labels = strat_train_set["median_house_value"].copy()

    housing = strat_train_set.drop("median_house_value", axis=1) 

    housing_ocen=housing.drop('ocean_proximity',axis=1)

    num_pipeline = Pipeline([        
                            ('imputer', Imputer(strategy="median")),        
                         ('attribs_adder', AttributeAdder()),        
                         ('std_scaler', StandardScaler()), 
                         
                         ])

    housing_num_tr = num_pipeline.fit_transform(housing_ocen)

    num_attribs = list(housing_ocen) 
    cat_attribs = ["ocean_proximity"]
    full_pipeline = ColumnTransformer([        
        ("num", num_pipeline, num_attribs),        
        ("cat", OneHotEncoder(), cat_attribs),    
        ])

    # X_train
    return full_pipeline.fit_transform(housing)

housing_prepared=fullPipline(housing)


#train the linear regression
from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(housing_prepared,housing_labels)


#some_data = housing.iloc[:5]
# predicted
house_predi=lin_reg.predict(housing_prepared)
from sklearn.metrics import mean_squared_error
lin_mse = mean_squared_error(housing_labels, house_predi) 
lin_rmse = np.sqrt(lin_mse) 
# this model mostly underfit thr data 
# solution is to find a more pewerfull model than this
from sklearn.tree import DecisionTreeRegressor
tr_reg=DecisionTreeRegressor()
tr_reg.fit(housing_prepared,housing_labels)
house_pred_tree=tr_reg.predict(housing_prepared)

tree_mse=mean_squared_error(housing_labels,house_pred_tree)
tree_rmse=np.sqrt(tree_mse)

from sklearn.model_selection import cross_val_score
score=cross_val_score(tr_reg,housing_prepared,housing_labels,cv=10,n_jobs=-1,
                      scoring="neg_mean_squared_error")
tree_rmse_score=np.sqrt(-score)

def display_result(scores):
    print( f"scores  {scores}")
    print(f" Mean {scores.mean()}")
    print(f"standar deviation {scores.std()}")

display_result(tree_rmse_score)

from sklearn.ensemble import RandomForestRegressor
ran=RandomForestRegressor()
ran.fit(housing_prepared,housing_labels)
score_ran=cross_val_score(ran,housing_prepared,housing_labels,cv=10,n_jobs=-1,
                      scoring="neg_mean_squared_error")

ran_rmse_score=np.sqrt(-score_ran)
display_result(ran_rmse_score)

from sklearn.model_selection import GridSearchCV
param=[
       {'n_estimators':[3,10,30],'max_features':[2,4,5,6,7,8]},
       {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]}
       ]
grid=GridSearchCV(ran, param, cv=5,                           
                  scoring='neg_mean_squared_error',                           
                  return_train_score=True)


grid.fit(housing_prepared, housing_labels)

grid.best_params_
grid.best_estimator_

cv_res=grid.cv_results_
for max_result,para in zip(cv_res["mean_test_score"],cv_res["params"]):
    print(np.sqrt(-max_result),para)
    
feature_importances = grid.best_estimator_.feature_importances_    
    
# test set prediction
final_model = grid.best_estimator_
X_test = strat_test_set.drop("median_house_value", axis=1) 
y_test = strat_test_set["median_house_value"].copy()

X_test_prepared = full_pipeline.transform(X_test)
final_predictions = final_model.predict(X_test_prepared)

final_mse = mean_squared_error(y_test, final_predictions) 
final_rmse = np.sqrt(final_mse)    
    
    
    
    
    
    
    
