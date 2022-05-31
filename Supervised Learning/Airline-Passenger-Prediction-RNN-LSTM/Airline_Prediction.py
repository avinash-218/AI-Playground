#given the number of passengers (in units of thousands) this month,
#what is the number of passengers next month?

#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

#import dataset
dataset = pd.read_csv('airline-passengers.csv')
#since first column-date which is separated by one month in each row,considering second column only
dataset = dataset.iloc[:,1].values
dataset = dataset.astype('float32')   #int -> float so that helpful in neural nets
dataset = dataset.reshape(144,1)

#visualise the data
plt.plot(dataset)
plt.xlabel('Time')
plt.ylabel('No.of Passengers')
plt.title('International Airline Passenger')
plt.show()

# fix random seed for reproducibility
np.random.seed(7)


#LSTMs are sensitive to the scale of the input data, specifically when the sigmoid
#(default) or tanh activation functions are used. It can be a good practice to rescale
#the data to the range of 0-to-1, also called normalizing.

#normalise
scaler = MinMaxScaler(feature_range=(0,1))
dataset = scaler.fit_transform(dataset.reshape(-1,1))

dataset = np.append(dataset[:-1],dataset[1:],axis=1)  #preprocessing dataset in which y(t)=x(t+1)

#train,test split (not random)
train,test = train_test_split(dataset,test_size=0.33,shuffle=False)
X_train = train[:,0]
Y_train = train[:,1]
X_test = test[:,0]
Y_test = test[:,1]

#reshaping to make in the format of LSTM (samples,timesteps,features)
X_train = X_train.reshape(X_train.shape[0],1,1)
X_test = X_test.reshape(X_test.shape[0],1,1)

#LSTM
model = Sequential()
model.add(LSTM(4,input_shape=(1,1)))
model.add(Dense(1))
model.compile(optimizer='adam',loss='mean_squared_error')
model.fit(X_train,Y_train,epochs=100,batch_size=1,verbose=1)

#predict
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

#inverse scaling all Ys to measure error
train_predict = scaler.inverse_transform(train_predict)
Y_train = scaler.inverse_transform(Y_train.reshape(-1,1))
test_predict = scaler.inverse_transform(test_predict)
Y_test = scaler.inverse_transform(Y_test.reshape(-1,1))

#calculate RMS
trainScore = math.sqrt(mean_squared_error(Y_train[:,0], train_predict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(Y_test[:,0], test_predict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))

# plot original and predictions (train)
plt.plot(Y_train)
plt.plot(train_predict,color='red')
plt.show()

# plot original and predictions (test)
plt.plot(Y_test)
plt.plot(test_predict,color='red')
plt.show()



