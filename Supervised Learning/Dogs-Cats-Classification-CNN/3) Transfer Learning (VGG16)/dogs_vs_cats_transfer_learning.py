import tensorflow as tf
import keras
import pandas as pd
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import models, optimizers
from tensorflow.keras.layers import Flatten, Dense, Dropout
#%%
#data preprocessing steps
# 1) read images
# 2) decode JPEG to RGB grid of pixels
# 3) convert to float values
# 4) rescale to interval [0,1]

train_dir = "../Datasets/train"
val_dir = "../Datasets/validation"
test_dir = "../Datasets/test"

train_data_gen = ImageDataGenerator(rescale = 1./255,
                                    rotation_range = 40,
                                    width_shift_range = 0.2,
                                    height_shift_range = 0.2,
                                    shear_range = 0.2,
                                    zoom_range = 0.2,
                                    horizontal_flip = True)
#Augmentation
#augmentation is done to reduce overfitting. So apply this only in training set
#rotation_range each image rotates in the range [-40degree,+40degree] - randomly chooses one degree in the range
#width_shift_range - shift images left or right by 20% of total width - randomly chooses one in the range
#height_shift_range  - shift images top or down by 20% of total height - randomly chooses one in the range
#similarly for shear and zoom
#horizontal flip 
                                    
test_data_gen = ImageDataGenerator(rescale=1./255)

train_generator = train_data_gen.flow_from_directory(
    train_dir,
    target_size = (150,150),
    batch_size = 20,
    class_mode = 'binary')

validation_generator = test_data_gen.flow_from_directory(
    val_dir,
    target_size = (150,150),
    batch_size = 20,
    class_mode = 'binary')
#%%
#transfer learning - download VGG16 convolutional base
from tensorflow.keras.applications import VGG16
#weights - imagenet is competition name
#include_top - False: only import convolution base, True: import entire model
#input_shape - our data set input shape
conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
print(conv_base.summary())
#%%
#visualise the convolutional base
tf.keras.utils.plot_model(conv_base, to_file='VGG16.png', show_shapes=True,show_layer_names=True, rankdir='TB', expand_nested=True)
#%%
#model creation
#layer1 - conv_base
#layer2 - flatten
#layer3 - Dense - 256 neurons
            #activations = 'relu'
#layer11 - Dense - 1 neurons activation='sigmoid'

model = models.Sequential()
model.add(conv_base)
model.add(Flatten())
model.add(Dropout(0.5)) #dropout - to reduce over fitting - deactivate 50% of neurons in each epoch randomly
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
print(model.summary())
#%%
#visualise model
tf.keras.utils.plot_model(model, to_file='Transfer_Model.png', show_shapes=True,show_layer_names=True, rankdir='TB', expand_nested=True)
#%%
#freeze conv_base (VGG16) layers non-trainable
conv_base.trainable = False
#if set to True , it takes around 8-9 hours for training
#now there are some non-trainable parameters
print(model.summary())
#%%
#compile
model.compile(loss='binary_crossentropy',optimizer = optimizers.RMSprop(lr=2e-5) , metrics=['accuracy'])
#%%
#early stopping
check_pt = keras.callbacks.ModelCheckpoint("Early_Stop.h5", save_best_only=True)
early_stop = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
#%%
#training

#Parameters:
#train_generator object of ImageDataGenerator
#steps_per_epoch - train_generator is generating data continuosly with batch_size=20 (i.e,20 images at a time)
    # so stopping point is to be given using this parameter.And train dataset is of 2000 images + augmented images
    # so it needs 2000//20 = 100 steps per epoch
#validation_data - is validation generator
#validation_steps - validation_generator is generating data continuosly with batch_size=20 (i.e,20 images at a time)
    # so stopping point is to be given using this parameter.And validation dataset is of 1000 images 
    # + augmented images so it needs 1000//20 = 50 steps per epoch
    
model_history = model.fit(train_generator,
                                    steps_per_epoch = 100,
                                    epochs = 30,
                                    validation_data = validation_generator,
                                    validation_steps = 50,
                                    callbacks=[check_pt, early_stop])
#%%
#visualise training
pd.DataFrame(model_history.history).plot()
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()
#%%
#testing
test_generator = test_data_gen.flow_from_directory(
    test_dir,
    target_size = (150,150),
    batch_size = 20,
    class_mode = 'binary')
print(model.evaluate(test_generator, steps=50))