import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import pickle

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

###KNN Classify###
# Calculating error/accuracy for K

knn = KNeighborsClassifier(n_neighbors=10)
start = datetime.datetime.now()
knn.fit(X_train, y_train)
endtrain = datetime.datetime.now()
print("Trainig Time =" + str(endtrain-start))
starttest = datetime.datetime.now()
pred_i = knn.predict(X_test)
endtest = datetime.datetime.now()
print("Test Time =" + str(endtest-starttest))

error= np.mean(pred_i != y_test)
acc = accuracy_score(y_test,pred_i)


print("Accuracy: " + str(acc*100))
print("Error: " + str(error*100))


# Save the Modle to file in the current working directory
Pkl_Filename = "Pickle_KNN_Model.pkl"

with open(Pkl_Filename, 'wb') as file:
    pickle.dump(knn, file)
# Load the Model back from file
with open(Pkl_Filename, 'rb') as file:
    Pickled_KNN_Model = pickle.load(file)

Pickled_KNN_Model




