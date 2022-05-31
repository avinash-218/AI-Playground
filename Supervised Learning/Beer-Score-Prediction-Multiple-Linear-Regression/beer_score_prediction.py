#%%
#import libraries
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
#%%
#import dataset
dataset = pd.read_csv('beer_data.csv')
dataset = dataset.iloc[:,:].values
#%%
#preprocess data
#convert cellar temparature range into two features (min temp and max_temp)
min_temp=[]
max_temp=[]
for i in dataset[:,2]:
    min_temp.append(int(i[0:2]))
    max_temp.append(int(i[3:]))
min_temp=np.array(min_temp)
max_temp=np.array(max_temp)
dataset = np.append(dataset,min_temp.reshape(-1,1),axis=1)
dataset = np.append(dataset,max_temp.reshape(-1,1),axis=1)
#removing cellar temperature range since two features was derived
dataset = np.delete(dataset,2,1)    
#swapping columns to make similar to the original dataset
dataset[:,[2,3]] = dataset[:,[3,2]] 
dataset[:,[4,3]] = dataset[:,[3,4]]
#%%
#train test split
X = dataset[:,:-1]
Y = dataset[:,-1]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,test_size=0.2,random_state=1)
Y_train = Y_train.reshape(-1,1)
Y_test = Y_test.reshape(-1,1)
#%%
#feature scaling
scale_x = StandardScaler()
X_train = scale_x.fit_transform(X_train)
X_test = scale_x.transform(X_test)
scale_y = StandardScaler()
#%%
#training
regressor = LinearRegression()
regressor.fit(X_train,scale_y.fit_transform(Y_train))
#%%
#evaluate with train data
h_train = regressor.predict(X_train)
h_train = scale_y.inverse_transform(h_train)
print(mean_squared_error(h_train,Y_train))
#%%
#test and evaluate with test data
h_test = regressor.predict(X_test)
h_test = scale_y.inverse_transform(h_test)
print(mean_squared_error(h_test,Y_test))