import datetime
import pickle

import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv('spotify_training_classification.csv')
df.sample(frac=1).reset_index(drop=True)
df.dropna(subset=[n for n in df if n != 'popularity_level'],how='all' ,inplace=True) #drop NaN rows

df.fillna(0)

features = ["acousticness", "danceability", "energy", "instrumentalness", "key", "liveness","duration_ms",
             "speechiness", "tempo", "valence"]

X = df[features]
Y = df['popularity_level']

X = X.replace((np.inf, -np.inf, np.nan), 0).reset_index(drop=True)
#train/test split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

###Adaboost classify###
bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),algorithm="SAMME",n_estimators=100)

start = datetime.datetime.now()
bdt.fit(X_train,y_train)
endtrain = datetime.datetime.now()
print("Trainig Time =" + str(endtrain-start))
starttest = datetime.datetime.now()
y_prediction = bdt.predict(X_test)
endtest = datetime.datetime.now()
print("Test Time =" + str(endtest-starttest))



accuracy=np.mean(y_prediction == y_test)*100
print("The achieved accuracy using Adaboost is " + str(accuracy))
error = []
clf = tree.DecisionTreeClassifier(max_depth=1)
clf.fit(X_train, y_train)
y_prediction = clf.predict(X_test)
accuracy=np.mean(y_prediction == y_test)*100
print("The achieved accuracy using Decision Tree is " + str(accuracy))

# Save the Model to file in the current working directory
Pkl_Filename = "Pickle_ADA_Model.pkl"

with open(Pkl_Filename, 'wb') as file:
    pickle.dump(bdt, file)
# Load the Model back from file
with open(Pkl_Filename, 'rb') as file:
    Pickled_ADA_Model = pickle.load(file)

Pickled_ADA_Model