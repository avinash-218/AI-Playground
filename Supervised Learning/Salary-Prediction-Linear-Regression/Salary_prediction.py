#%%
#import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
#%%
#import dataset
dataset=pd.read_csv("Salary_Data.csv")
dataset = dataset.iloc[:,:].values
#%%
#train test split
X = dataset[:,0]
Y = dataset[:,0]
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=1)
#%%
#train
regressor = LinearRegression()
regressor.fit(X_train.reshape(-1,1),Y_train.reshape(-1,1))
#evaluate model using trained data
h_train = regressor.predict(X_train.reshape(-1,1))
print(mean_squared_error(Y_train,h_train))
#visualise
plt.scatter(X_train,Y_train,color='r')
plt.plot(X_train,h_train,color='b')
#%%
#evaluate model using test data
h_test = regressor.predict(X_test.reshape(-1,1))
print(mean_squared_error(h_test,Y_test))
#visulaise
plt.scatter(X_test,Y_test,color='r')
plt.plot(X_test,h_test,color='b')