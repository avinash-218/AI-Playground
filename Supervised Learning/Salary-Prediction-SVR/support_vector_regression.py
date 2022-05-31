#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#import dataset
dataset=pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:-1].values
Y = dataset.iloc[:,-1].values

#feature scaling
Y = Y.reshape(-1,1)   #reshape into 2darray
from sklearn.preprocessing import StandardScaler
scale_X = StandardScaler()
scale_Y = StandardScaler()
X = scale_X.fit_transform(X)
Y = scale_Y.fit_transform(Y)

#train model
from sklearn.svm import SVR
regressor = SVR()
regressor.fit(X, Y.ravel())    #reshape into compatible

#predict
pred_val = np.array([6.5])
pred_val = pred_val.reshape(1,1)    #reshape into compatible
pred_val = scale_X.transform(pred_val)   #scaling
Y_pred = regressor.predict(pred_val)
Y_pred = scale_Y.inverse_transform(Y_pred)

#visualisation
plt.scatter(scale_X.inverse_transform(X),scale_Y.inverse_transform(Y),color='r')
plt.plot(scale_X.inverse_transform(X),scale_Y.inverse_transform(regressor.predict(X)))
plt.title('Truth or Bluff')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

#higher resolution and smooth curve SVM
X_grid = np.arange(min(scale_X.inverse_transform(X)),max(scale_X.inverse_transform(X)),0.1)
X_grid = X_grid.reshape(-1,1)
plt.scatter(scale_X.inverse_transform(X),scale_Y.inverse_transform(Y),color='r')
plt.plot(X_grid,scale_Y.inverse_transform(regressor.predict(scale_X.transform(X_grid))),color='b')
plt.title('Truth or Bluff')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()