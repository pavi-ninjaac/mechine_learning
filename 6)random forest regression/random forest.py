#Random forest Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv(r'C:\Users\ninjaac\Desktop\P14-Machine-Learning-AZ-Template-Folder\Part 2 - Regression\Section 8 - Decision Tree Regression\Decision_Tree_Regression\Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values
#if you add more no of trees ypu wont have the more number of steps you have mo
#more value to avrage for the prediction
#(random forest is the forest of trees)
#first get the K random vales
#then make decision tree on the K values
#make N no of threes by n_estimator 
#and get average of the all trees prediction as the prediction
# Fitting Multiple Linear Regression to the Training set


from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=1000, random_state=0)
regressor.fit(X,y)

# Predicting the Test set results
y_pred = regressor.predict([[6.5]])

#normal visualization

plt.scatter(X,y,color='red')
plt.plot(X,regressor.predict(X),color='blue')
plt.xlabel('levels')
plt.ylabel('salaries')
plt.title('level vs salries')
plt.show()
#visualization 

X_grid=np.arange(min(X),max(X),0.1)
X_grid=X_grid.reshape(len(X_grid),1)
plt.scatter(X,y,color='red')
plt.plot(X_grid,regressor.predict(X_grid),color='blue')
plt.xlabel('levels')
plt.ylabel('salaries')
plt.title('level vs salries')
plt.show()
