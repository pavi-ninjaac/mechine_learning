# -*- coding: utf-8 -*-
"""
time:22-04-2020 23-18

author:@pavi ninjac
"""

# import the neccesary files 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import the dataset using pandas
dataset=pd.read_csv(r'C:\Users\ninjaac\Desktop\P14-Machine-Learning-AZ-Template-Folder\books\taks\classification_3edchapter\emails.csv\emails.csv')

# find the shape
np.shape(dataset) #(5728,2)
dataset.columns # text,spam

# checking for the duplication in the dataset
dataset.drop_duplicates(inplace=True)
print(f"dataset shape after duplication drop {np.shape(dataset)}") #dataset shape after duplication drop (5695, 2)
pd.DataFrame(dataset.isnull().sum()) #text  0 spam  0

#using Natural Language Processing to cleaning the dataset for prediction
import nltk
import re
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus=[]

#delete the charecter other than the alphapets
email=re.sub('[^a-zA-Z]'," ",dataset["text"][0])
email=email.lower()
# the sting is converted to a list of words
email=email.split()
len(email) # 188 words before stemming
#stemming removing the tens in the words
ps=PorterStemmer()

email=[ps.stem(word) for word in email if not word in set(stopwords.words('english'))]
len(email)# 133 words after stemming that means 5 stopswords are removed
#join those words using the join function ofstrinng

email=" ".join(email)
corpus.append(email)


