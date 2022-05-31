import tensorflow as tf
import keras
import pandas as pd
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import models, optimizers
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
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
    batch_size = 32,
    class_mode = 'binary')

validation_generator = test_data_gen.flow_from_directory(
    val_dir,
    target_size = (150,150),
    batch_size = 32,
    class_mode = 'binary')
#%%
#model creation
#layer1 - conv - 32 filters, kernel_size =(3,3), input_shape = 150*150*3 (150- breadth&width;3-RGB)
#layer2 - maxpool - 2x2
#layer3 - conv - 64 filters, kernel-size=(3,3)
#layer4 - maxpool - 2x2
#layer5 - conv - 128 filters, kernel-size=(3,3)
#layer6 - maxpool - 2x2
#layer7 - conv - 128 filters, kernel-size=(3,3)
#layer8 - maxpool - 2x2
#layer9 - flatten
#layer10 - Dense - 512 neurons
            #activations = 'relu'
#layer11 - Dense - 1 neurons activation='sigmoid'

model = models.Sequential()
model.add(Conv2D(filters=32, input_shape=(150, 150, 3), kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Flatten())
model.add(Dropout(0.5)) #dropout - to reduce over fitting - deactivate 50% of neurons in each epoch randomly
model.add(Dense(512, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
print(model.summary())
#%%
#visualise model
tf.keras.utils.plot_model(model, to_file='Model.png', show_shapes=True,show_layer_names=True, rankdir='TB', expand_nested=True)
#%%
#compile
model.compile(loss='binary_crossentropy',optimizer = optimizers.RMSprop(lr=1e-4) , metrics=['accuracy'])
#%%
#early stopping
check_pt = keras.callbacks.ModelCheckpoint("Early_Stop.h5", save_best_only=True)
early_stop = keras.callbacks.EarlyStopping(patience = 10, restore_best_weights=True)
#%%
#training

#Parameters:
#train_generator object of ImageDataGenerator
#steps_per_epoch - train_generator is generating data continuosly with batch_size=32 (i.e,32 images at a time)
    # so stopping point is to be given using this parameter.And train dataset is of 2000 images + augmented images
    # so it needs 2000//32 = 62 steps per epoch
#validation_data - is validation generator
#validation_steps - validation_generator is generating data continuosly with batch_size=32 (i.e,32 images at a time)
    # so stopping point is to be given using this parameter.And validation dataset is of 1000 images 
    # + augmented images so it needs 1000//32 = 31 steps per epoch
    
model_history = model.fit(train_generator,
                                    steps_per_epoch = 62,
                                    epochs = 100,
                                    validation_data = validation_generator,
                                    validation_steps = 31,
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