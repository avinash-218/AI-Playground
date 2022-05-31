# Bank-Customer-Retainment-Prediction-ANN

This deep learning model is used to predict if a customer of a bank retains as the customer or leave base on given data sets in Churn modelling.csv.


1) Data preprocessing (categorical encoding and one hot encoding)
2) Data splitting (train and test set)
3) Feature Scaling
4) HyperParameter Tuning

Note: run hyperparameter tuning once to get the best parameters and then use the best parameters.

CPU:
#best parameters: batch_size=10,epochs=500,optimizer=rmsprop (accuracy=0.854)
#best parameters: batch_size=25,epochs=500,optimizer=rmsprop (accuracy=0.85063)

GPU:
#best parameters: batch_size=10,epochs=500,optimizer=rmsprop (accuracy=0.8685)
