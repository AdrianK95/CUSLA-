# -*- coding: utf-8 -*-
"""
algorithm to predict the star rating of a users interaction with a "dumb" chatbot
"""

import csv
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap

import pandas as pd
#reading in data from csv #assigning column headers # no index column
df = pd.read_csv("data_chatCombined.csv",
                 names=["messages","time","sentiment","goodbye","stars"], index_col=False)

#creating an array of the data
X = np.array([df["messages"],df["time"],df["sentiment"],df["goodbye"]])
#transoposing the array for matrix mulplication
X =X.transpose()

#setting our target data to try to predict
Y = np.array(df["stars"])

#printing the shaped of the arrays
print(X.shape)
print(Y.shape)

#creating the training and test sets
X_train,X_test, Y_train,Y_test = train_test_split(X,Y, test_size = .2, random_state=0)

#number of clusters to be made
clf = KNeighborsClassifier(n_neighbors =5)

#fitting the data
clf.fit(X_train,Y_train)

#test set predictions and accuracy of model
print("Test set predictions: {}".format(clf.predict(X_test)))
print("Test set accuracy: {}".format(clf.score(X_test,Y_test)))

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = np.array(df['messages'])
y = np.array(df['sentiment'])
z = np.array(df['goodbye'])

ax.scatter(x,y,z, marker="s", c=df["stars"], s=40, cmap="RdBu")

plt.show()

X_new = np.array([[12,130,1,0]])
print(clf.predict(X_new))
