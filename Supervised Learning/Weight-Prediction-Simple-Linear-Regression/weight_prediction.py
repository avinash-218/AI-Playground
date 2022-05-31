#%%
#import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
#%%
#import dataset
dataset = pd.read_csv('data.csv')
dataset = dataset.iloc[:,:].values
X = dataset[:,0]
Y = dataset[:,0]
#%%
#visualise data
plt.scatter(X,Y)
plt.title('Height Vs Weight')
#%%
regressor = LinearRegression()
regressor.fit(X.reshape(-1,1),Y.reshape(-1,1))
#%%
Y_pred = regressor.predict(X.reshape(-1,1))
print(mean_squared_error(Y,Y_pred))
#%%
#visualise the result
plt.scatter(X,Y,color='r')
plt.plot(X,Y_pred,color='b')
plt.show()