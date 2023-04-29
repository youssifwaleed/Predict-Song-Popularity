import pickle

import numpy as np
import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from LabelEncoder import *



#Load songs data
data = pd.read_csv('spotify_training.csv')
#Drop the rows that contain missing values
data.dropna(how='any',inplace=True)

#data.drop(labels='name',axis=1)
song_data=data.iloc[:,:]
X=data.iloc[:,0:17] #Features
Y=data['popularity'] #Label
cols=('artists','id','instrumentalness','name','release_date')
start = time.time()
X=Feature_Encoder(X,cols);


#Split the data to training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.30,shuffle=True)
#Get the correlation between the features
corr = song_data.corr()
#Top 50% Correlation training features with the Value
top_feature = corr.index[abs(corr['popularity']>0.5)]
#Correlation plot
plt.subplots(figsize=(12, 8))
top_corr = song_data[top_feature].corr()
sns.heatmap(top_corr, annot=True)
plt.show()

poly_features = PolynomialFeatures(degree=2)

# transforms the existing features to higher degree features.
X_train_poly = poly_features.fit_transform(X_train)

# fit the transformed features to Linear Regression
poly_model = linear_model.LinearRegression()
poly_model.fit(X_train_poly, y_train)

# predicting on training data-set
y_train_predicted = poly_model.predict(X_train_poly)

# predicting on test data-set
prediction = poly_model.predict(poly_features.fit_transform(X_test))
stop = time.time()

print(f"Training time: {stop - start}s")
print('Co-efficient of Polynomial regression',poly_model.coef_)
print('Intercept of Polynomial regression model',poly_model.intercept_)
print('Mean Square Error', metrics.mean_squared_error(y_test, prediction))

true_song_value=np.asarray(y_test)[0]
predicted_song_value=prediction[0]
print('True value for the first player in the test set  : ' + str(true_song_value))
print('Predicted value for the first player in the test set  : ' + str(predicted_song_value))

# Save the Modle to file in the current working directory
Pkl_Filename = "Pickle-Poly-Model.pkl"

with open(Pkl_Filename, 'wb') as file:
    pickle.dump(poly_model, file)
# Load the Model back from file
with open(Pkl_Filename, 'rb') as file:
    Pickled_poly_Model = pickle.load(file)

Pickled_poly_Model

