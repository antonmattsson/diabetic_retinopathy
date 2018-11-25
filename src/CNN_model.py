
# coding: utf-8

# In[2]:


import numpy as np
from image_generator import *
from numpy.core.defchararray import add, replace
from keras.layers import Flatten, Dense, ZeroPadding2D, Conv2D, Activation, MaxPooling2D, Dropout, BatchNormalization
from keras.models import Sequential

          


# In[3]:


# Set the number of training samples
n_train = 60
# Read filenames from a text file listing all the images
filenames = np.genfromtxt('../data/train_list.txt', dtype=str)[:n_train]
# Add path of the data folder to the files
filepaths = add(np.full(shape=(filenames.shape), fill_value='../data/train/'), filenames)

# Remove .jpeg from the end of file names to search from the labels
train_samples = replace(filenames, ".jpeg", "")
train_labels = np.genfromtxt('../data/trainLabels.csv', skip_header=1, dtype=str, delimiter=',')
# Choose labels for the chosen training images only
mask = np.isin(train_labels[:,0],train_samples)
sample_labels = train_labels[mask, 1]

# Set batch size, image shape and patch size
batch_size = 12
img_shape = (512, 512)

# Create an instance of the image generator
train_gen = ImageGenerator(filepaths, sample_labels, batch_size, img_shape)


#add convolutional network model

model = Sequential()
model.add(Conv2D(64, kernel_size=(4, 4), input_shape=(img_shape[0], img_shape[1], 3)))
model.add(Activation('relu'))


model.add(Conv2D(64, kernel_size=(4, 4), padding='same'))
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(4, 4), padding='same'))
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Activation('relu'))

model.add(Dense(64, activation="relu"))

model.add(Dropout(0.5))
model.add(Dense(5, activation="softmax"))

#compile and fit model

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit_generator(generator=train_gen,
                    steps_per_epoch=(n_train // batch_size),
                    epochs=20, verbose=1)

#print model summary
model.summary()

