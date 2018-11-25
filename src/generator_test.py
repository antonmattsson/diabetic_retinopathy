import numpy as np
from image_generator import *
from numpy.core.defchararray import add, replace
from keras.layers import Flatten, Dense
from keras.models import Sequential

# Set the number of training samples
n_train = 60
# Read filenames from a text file listing all the images
filenames = np.genfromtxt('../data/train_filenames.txt', dtype=str)[:n_train]
# Add path of the data folder to the files
filepaths = add(np.full(shape=(filenames.shape), fill_value='../data/train/'), filenames)

# Remove .jpeg from the end of file names to search from the labels
train_samples = replace(filenames, ".jpeg", "")
train_labels = np.genfromtxt('../data/trainLabels.csv', skip_header=1, dtype=str, delimiter=',')
# Choose labels for the chosen training images only
mask = np.isin(train_labels[:,0],train_samples)
sample_labels = train_labels[mask, 1]

# Set batch size and image shape
batch_size = 12
img_shape = (400, 400)

# Create an instance of the image generator
train_gen = ImageGenerator(filepaths, sample_labels, batch_size, img_shape)

# Define the most simple keras model
model = Sequential()
model.add(Flatten(input_shape=(img_shape[0], img_shape[1], 3)))
model.add(Dense(5, activation="softmax"))

# Compile and fit the model
model.compile(optimizer='adam', loss='categorical_crossentropy')

model.fit_generator(generator=train_gen,
                    steps_per_epoch=(n_train // batch_size),
                    epochs=2,
                    verbose=1)
