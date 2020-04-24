# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 22:55:12 2020

@author: ninjaac
"""


#importing dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#import dataset 
dataset=pd.read_csv(r'C:\Users\ninjaac\Desktop\warehousing-assigmnent\auto_data.csv',header=None)

np.shape(dataset) #(205, 26)

# remove the rows that containing the '?' ti will return the data that containig the 
#dataframe removed by that rows
dataset.drop(dataset[dataset.values=='?'].index,inplace=True)
np.shape(dataset) #(159,26)
# contain 159 instance after drop those samples
# drop the duplicate values and none values
dataset.drop_duplicates(inplace=True)
np.shape(dataset) # (159,26) nothing is duplicate
dataset.dropna(how="any")
np.shape(dataset) # (159, 26) no none values
# creating pipeline for data cleaning 
