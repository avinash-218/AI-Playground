import pandas as pd
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import models, optimizers
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
#%%
#data preprocessing steps
# 1) read images
# 2) decode JPEG to RGB grid of pixels
# 3) convert to float values
# 4) rescale to interval [0,1]

#paths of dataset
train_dir = "Datasets/train"
val_dir = "Datasets/validation"
test_dir = "Datasets/test"

train_data_gen = ImageDataGenerator(rescale=1./255)
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
model.add(Dense(512, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
print(model.summary())
#%%
#compile
model.compile(loss='binary_crossentropy',optimizer = optimizers.RMSprop(lr=1e-4) , metrics=['accuracy'])
#%%
#training

#Parameters:
#train_generator object of ImageDataGenerator
#steps_per_epoch - train_generator is generating data continuosly with batch_size=20 (i.e,20 images at a time)
    # so stopping point is to be given using this parameter.And train dataset is of 2000 images
    # so it takes 2000/20 steps
#validation_data - is validation generator
#validation_steps is similar to steps_per_epoch except this is for validation data
    #validation_generator generates data continuously with batch_size =20 (i.e, 20 images at a time)
    #so stopping point is to be given using this parameter. And validation dataset is of 1000 images
    # so it takes 1000/20 steps

model_history = model.fit_generator(train_generator,
                                    steps_per_epoch = 100,
                                    epochs = 20,
                                    validation_data = validation_generator,
                                    validation_steps = 50)
#%%
#visualise training
pd.DataFrame(model_history.history).plot()
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()
#%%
#save model
model.save("Model_Without_Augmentation.h5")
#%%
#testing
test_generator = test_data_gen.flow_from_directory(
    test_dir,
    target_size = (150,150),
    batch_size = 20,
    class_mode = 'binary')
print(model.evaluate(test_generator, steps=50))
#this is overfitted model data augmentation to be done
