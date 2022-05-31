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
#reshaping to adapt CNN model
X_train = X_train.reshape((60000, 28, 28, 1))
X_test = X_test.reshape((10000, 28, 28, 1))
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
#input layer - 28*28*1 pixel image
#conv layer - 26*26*32    filter size -3*3  32 filters  stride - 1   padding - valid (ignore a column in left and right)
#pooling layer - 13*13*32   2*2 - max pooling 
#flatten - 5408
#two hidden layers with RELU activation -300 and 100 neurons respectively in layers
#output layer - 10 classes with softmax activation 
#%%
#building model
model = keras.models.Sequential()
model.add(keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=1, padding='valid', activation='relu', input_shape=(28, 28, 1)))
model.add(keras.layers.MaxPooling2D((2, 2)))

model.add(keras.layers.Flatten())   #flatten
model.add(keras.layers.Dense(300, activation = 'relu'))
model.add(keras.layers.Dense(100, activation = 'relu'))
model.add(keras.layers.Dense(10, activation='softmax'))
print(model.summary())
#%%
#visualise model
from keras.utils import plot_model
plot_model(model, to_file='model_plot1.png',show_shapes=True)
#%%
#learning
#since Y is labels using sparse_categorical_crossentropy (if Y is probability of class use categorical_crossentropy)
#if Y is T/F use binary_crossentropy
model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd", metrics=['accuracy'])
#%%
#setting checkpoints
checkpts = keras.callbacks.ModelCheckpoint("Early_Stopping-Save.h5")
early_stop = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
#patience - model will stop training only after for 10 epochs val accuracy is not increasing
#%%
#training
model_history = model.fit(X_train, Y_train, epochs=300, batch_size=64, validation_data=(X_valid, Y_valid), callbacks=[checkpts, early_stop] )
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
plt.imshow(X_test[1])
print(Y_pred_class[1])
#%%

