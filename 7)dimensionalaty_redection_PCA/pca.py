# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#confision matrix
from sklearn.metrics import confusion_matrix

# Importing the dataset
dataset = pd.read_csv(r'C:\Users\ninjaac\Desktop\P14-Machine-Learning-AZ-Template-Folder\Part 9 - Dimensionality Reduction\Section 43 - Principal Component Analysis (PCA)\PCA\Wine.csv')
X = dataset.iloc[:,0:13].values
y = dataset.iloc[:, 13].values


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)

#applying PCA
from sklearn.decomposition import PCA
pca=PCA(n_components=2)
X_train=pca.fit_transform(X_train)
X_test=pca.fit_transform(X_test)
#that will give you the top to bottom order of correlation of indepented varibles that i can decompose
explained_variance=pca.explained_variance_ratio_

# Fitting logistric Regression to the Training set
from sklearn.linear_model  import LogisticRegression
classi = LogisticRegression(random_state=0)
classi.fit(X_train,y_train)

# Predicting the Test set results
y_pred = classi.predict(X_train)
y_pr=classi.predict(X_test)
con_max=confusion_matrix(y_test, y_pr)

#train set visualization

from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classi.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green','blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green','blue'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('PCA!')
plt.ylabel('PCA2')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classi.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green','blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green','blue'))(i), label = j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.legend()
plt.show()
