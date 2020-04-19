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














