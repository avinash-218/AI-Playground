#importing packages and libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#import dataset
dataset=pd.read_csv('Position_Salaries.csv')
X= dataset.iloc[:,-2].values
Y= dataset.iloc[:,-1].values

#train linear regression
from sklearn.linear_model import LinearRegression
linear_regression = LinearRegression()
linear_regression.fit(X.reshape(-1,1),Y)

#train polynomial regression (degree2)
from sklearn.preprocessing import PolynomialFeatures
poly_features2 = PolynomialFeatures(degree=2)
X_poly2 = poly_features2.fit_transform(X.reshape(-1,1))
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly2,Y)

#train polynomial linear regression with higher degree (degree 3)
from sklearn.preprocessing import PolynomialFeatures
poly_features3 = PolynomialFeatures(degree=3)
X_poly3 = poly_features3.fit_transform(X.reshape(-1,1))
lin_reg_3 = LinearRegression()
lin_reg_3.fit(X_poly3,Y)

#train polynomial linear regression with higher degree (degree 4)
from sklearn.preprocessing import PolynomialFeatures
poly_features4 = PolynomialFeatures(degree=4)
X_poly4 = poly_features4.fit_transform(X.reshape(-1,1))
lin_reg_4 = LinearRegression()
lin_reg_4.fit(X_poly4,Y)

#visualise linear regression
plt.scatter(X,Y)
plt.plot(X,linear_regression.predict(X.reshape(-1,1)))
plt.show()

#visulaise polynomial linear regression (degree 2)
plt.scatter(X,Y)
plt.plot(X,lin_reg_2.predict(X_poly2))
plt.show()

#visulaise polynomial linear regression (degree 3)
plt.scatter(X,Y)
plt.plot(X,lin_reg_3.predict(X_poly3))
plt.show()

#visulaise polynomial linear regression (degree 4)
plt.scatter(X,Y)
plt.plot(X,lin_reg_4.predict(X_poly4))
plt.show()

#predict salary for employee with level 6.5

#prediction with linear regression
print('Degree 1 :',linear_regression.predict([[6.5]]))

#prediction with polynomial linear regression (degree 2)
print('Degree 2 :',lin_reg_2.predict(poly_features2.fit_transform([[6.5]])))

#prediction with polynomial linear regression (degree 3)
print('Degree 3 :',lin_reg_3.predict(poly_features3.fit_transform([[6.5]])))

#prediction with polynomial linear regression (degree 4)
print('Degree 4 :',lin_reg_4.predict(poly_features4.fit_transform([[6.5]])))

