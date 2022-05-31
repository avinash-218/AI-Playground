#import libraries
import numpy as np
import pandas as pd

#import dataset
dataset=pd.read_csv('50_Startups.csv')
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,-1].values

#encode categorical data
from sklearn.preprocessing import OneHotEncoder
one_hot_encoder = OneHotEncoder()
X = np.column_stack((one_hot_encoder.fit_transform(X[:,-1].reshape(-1,1)).toarray(),X[:,:-1]))

#train test split
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)

#training
from sklearn.linear_model import LinearRegression
linear_regression = LinearRegression()
linear_regression.fit(X_train,Y_train)

#predict
Y_pred = linear_regression.predict(X_test)

#accuracy
from sklearn.metrics import mean_squared_error
import math
accuracy = math.sqrt(mean_squared_error(Y_pred,Y_test))/100
print("Accuracy : {}".format(round(accuracy,2)))

#regression equation
print("\nCoefficients :\n",linear_regression.coef_)
print("\nIntercept :",linear_regression.intercept_)

print('\nRegression Equation:')
print("\nProfit = 86.6×Dummy State 1 − 873×Dummy State 2 + 786×Dummy State 3 \n − 0.773×R&D Spend + 0.0329×Administration + 0.0366×Marketing Spend + 42467.53")