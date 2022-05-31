#import packages
import numpy as np
import matplotlib as plt
import pandas as pd

#data preprocessing

#import dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X=dataset.iloc[:,3:13].values
Y=dataset.iloc[:,13].values

#label encode categorical variables (Geography ,gender)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label_encoder_X_1 = LabelEncoder()
X[:,1]=label_encoder_X_1.fit_transform(X[:,1])
label_encoder_X_2 = LabelEncoder()
X[:,2]=label_encoder_X_2.fit_transform(X[:,2])

#one hot encode label encoded categorical variable
#since gender is 0 or 1 we can consider it as male or not i.e,0-female 1-male or vice versa
#but since geography can be 0,1,2 and also they are just category and not of higher value 
#than other, so we should create new feature out of the label encoded feature
one_hot_encoder = OneHotEncoder()
X=np.column_stack((one_hot_encoder.fit_transform(X[:,1].reshape(-1,1)).toarray(),X))

#remove the original geography which is column 4
X=np.delete(X,4,1)

#like gender we can find the geography from two columns instead of three columns
#so we remove first column (any can be removed) to avoid dummy variable trap (redundancy)
X=X[:,1:]

#split dataset to train and test data
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=0 )

#feature scaling is to be done because we do not need one feature to dominate other
from sklearn.preprocessing import StandardScaler
f_scale=StandardScaler()
X_train = f_scale.fit_transform(X_train)
X_test = f_scale.transform(X_test)

#tuning ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
import keras
from keras.models import Sequential   #initialisation the ANN
from keras.layers import Dense   #build ANN

def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(input_dim=11, units=6,activation='relu',kernel_initializer='uniform'))
    classifier.add(Dense(units=6, activation='relu',kernel_initializer='uniform'))
    classifier.add(Dense(units=1, activation='sigmoid',kernel_initializer='uniform'))
    classifier.compile(optimizer=optimizer,loss='binary_crossentropy', metrics=['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn=build_classifier) #not build_classifier() (no brackets)

parameters = {'batch_size':[10, 25, 32],
              'epochs':[100, 500],
              'optimizer':['adam', 'rmsprop']}

grid_search = GridSearchCV(estimator=classifier, param_grid=parameters, scoring = 'accuracy', cv= 10)
grid_search = grid_search.fit(X_train, Y_train)

best_parameters = grid_search.best_params_
best_accuracy= grid_search.best_score_
    
#best parameters: batch_size=10,epochs=500,optimizer=rmsprop (0.854)
#best parameters: batch_size=25,epochs=500,optimizer=rmsprop (0.85063)
