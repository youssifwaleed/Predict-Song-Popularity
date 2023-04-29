import datetime
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('spotify_training_classification.csv')

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

###RForest Classify###
RFC_Model = RandomForestClassifier(n_estimators=50, max_leaf_nodes=120)
start = datetime.datetime.now()
RFC_Model.fit(X_train, y_train)
endtrain = datetime.datetime.now()
print("Trainig Time =" + str(endtrain-start))
starttest = datetime.datetime.now()
RFC_Predict = RFC_Model.predict(X_test)
endtest = datetime.datetime.now()
print("Test Time =" + str(endtest-starttest))

RFC_Accuracy = accuracy_score(y_test, RFC_Predict)
print("Accuracy: " + str(RFC_Accuracy*100))

# Save the Modle to file in the current working directory
Pkl_Filename = "Pickle_RF_Model.pkl"

with open(Pkl_Filename, 'wb') as file:
    pickle.dump(RFC_Model, file)
# Load the Model back from file
with open(Pkl_Filename, 'rb') as file:
    Pickled_RF_Model = pickle.load(file)

Pickled_RF_Model