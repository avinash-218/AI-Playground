#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
#%%

#training
dataset = pd.read_csv('train.csv')

#data cleaning
dataset['Age'].fillna(dataset['Age'].median(), inplace=True)

#encoding
sex = pd.get_dummies(dataset['Sex'], prefix='Sex')
embarked = pd.get_dummies(dataset['Embarked'], prefix='Embarked')
Pclass = pd.get_dummies(dataset['Pclass'], prefix='Pclass')
dataset = pd.concat([dataset,sex, embarked, Pclass], axis=1)
dataset = dataset.drop(['PassengerId', 'Pclass', 'Name', 'Sex', 'Embarked', 'Ticket', 'Cabin'] , axis =1)

#train-test-split
X = dataset.drop('Survived', axis=1)
Y = dataset['Survived']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 4)

#training
rfc = RandomForestClassifier(n_estimators=15)   #15 trees
rfc.fit(X_train, Y_train)

#train accuracy
y_pred = rfc.predict(X_train)
print("train accuracy -", accuracy_score(Y_train, y_pred)*100)

#test accuracy
y_pred = rfc.predict(X_test)
print("test accuracy -", accuracy_score(Y_test, y_pred)*100)

#%%
#prediction
pred_set = pd.read_csv('test.csv')
#data cleaning
pred_set['Age'].fillna(pred_set['Age'].median(), inplace=True)
pred_set['Fare'].fillna(pred_set['Fare'].median(), inplace=True)

#encoding
sex = pd.get_dummies(pred_set['Sex'], prefix='Sex')
embarked = pd.get_dummies(pred_set['Embarked'], prefix='Embarked')
Pclass = pd.get_dummies(pred_set['Pclass'], prefix='Pclass')
pred_set = pd.concat([pred_set,sex, embarked, Pclass], axis=1)
pred_set = pred_set.drop(['PassengerId', 'Pclass', 'Name', 'Sex', 'Embarked', 'Ticket', 'Cabin'] , axis =1)

#predict values
y_pred = rfc.predict(pred_set)