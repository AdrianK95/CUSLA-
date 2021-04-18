import csv

from sklearn.model_selection import train_test_split
from sklearn.neural_network import  MLPClassifier
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap

with open("data_chat (1).csv") as csvfile:
    data = csv.DictReader(csvfile, fieldnames=("nmessages","time","sentiment", "goodbye", "stars"))
    print(data)


import pandas as pd
df = pd.read_csv("data_chat (1).csv",
                 names=["messages","time","sentiment","goodbye","stars"], index_col=False)

X = np.array([df["messages"],df["time"],df["sentiment"],df["goodbye"]])
X =X.transpose()
Y = np.array(df["stars"])

print(X.shape)
print(Y.shape)

X_train,X_test, Y_train,Y_test = train_test_split(X,Y, random_state=0)


mlp = MLPClassifier(activation = 'relu', solver='adam',hidden_layer_sizes=[100], max_iter=1000).fit(X_train, Y_train)


print("Test set predictions: {}".format(mlp.predict(X_test)))
print("Test set accuracy: {}".format(mlp.score(X_test,Y_test)))

