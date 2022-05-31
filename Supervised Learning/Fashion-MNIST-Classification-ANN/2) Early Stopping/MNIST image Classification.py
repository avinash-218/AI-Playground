#import libraries
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import keras
import tensorflow as tf
#%%
#import dataset
fashion_mnist = keras.datasets.fashion_mnist
(X_train, Y_train), (X_test, Y_test) = fashion_mnist.load_data()
#%%
#setting labels
class_names =np.array(["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"])
#%%
#view data
plt.imshow(X_train[10])  #imshow - image show
print(class_names[Y_train[10]])
#%%
#normalisation (by 255 since grayscale is 0 to 255)
X_train = X_train /255.0
X_test = X_test /255.0
#%%
#train set to train and validation set
#validation set model optimization
X_valid, X_train = X_train[:5000], X_train[5000:]
Y_valid, Y_train = Y_train[:5000], Y_train[5000:]
#%%
#set randomseed as 42
np.random.seed(42)
tf.random.set_seed(42)
#%%
#model description
#input layer - 28*28 pixel image
#two hidden layers with RELU activation -300 and 100 neurons respectively in layers
#output layer - 10 classes with softmax activation 
#%%
#building model
model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=(28, 28)))   #flatten the images in input layer
model.add(keras.layers.Dense(300, activation = 'relu'))
model.add(keras.layers.Dense(100, activation = 'relu'))
model.add(keras.layers.Dense(10, activation='softmax'))
print(model.summary())
#%%
#visualise model
from keras.utils import plot_model
plot_model(model, to_file='model_plot1.png',show_shapes=True)
#%%
#get weights
w, b = model.layers[1].get_weights()
#%%
#learning
#since Y is labels using sparse_categorical_crossentropy (if Y is probability of class use categorical_crossentropy)
#if Y is T/F use binary_crossentropy
model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd", metrics=['accuracy'])
#%%
#early stopping
checkpts = keras.callbacks.ModelCheckpoint("Early_Stop-Save.h5", save_best_only=True)
#save_best_only - saves only model with best val accuracy
early_stop = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
#patience - stop model training if val accuracy is not improving for 10 epochs continuously
#%%
#training
model_history = model.fit(X_train, Y_train, epochs=500, validation_data=(X_valid, Y_valid), callbacks=[checkpts, early_stop])
#%%
#visualising the training
pd.DataFrame(model_history.history).plot()
plt.grid(True)
plt.gca().set_ylim(0,1)
plt.show()
#%%
#model evaluation
eval_hist = model.evaluate(X_test, Y_test)
#first value - loss
#second value - accuracy
#%%
#prediction
Y_pred = model.predict(X_test)  #probability of the image to belong in the classes
#Y_pred = Y_pred.round(2)        #round the probability to two places
Y_pred_class = model.predict_classes(X_test)
Y_pred_class = class_names[Y_pred_class]
#%%
#verifying manually
plt.imshow(X_test[0])
print(Y_pred_class[0])
#%%
