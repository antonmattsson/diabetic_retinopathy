import numpy as np
from keras.layers import Flatten, Dense, ZeroPadding2D, Conv2D, Activation, MaxPooling2D, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Flatten, Dense, Conv2D, Activation, MaxPooling2D, Dropout
from keras.models import Sequential, load_model
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
import pickle
from performance_plots import *
from pathlib import Path

# Set the number of training samples
train_steps = 400
validation_steps = 75 # maximum
batch_size = 32
img_shape = (256, 256)

# Set data augmentation protocol
datagen = ImageDataGenerator(
        rotation_range=40,
        shear_range=0.2,
        zoom_range=0.2,
        rescale=1./255,
        horizontal_flip=True,
        fill_mode='nearest')

# Construct generators for the training and validation data using data augmentation
train_gen = datagen.flow_from_directory('../data/augmentation_train',
                                        target_size=(256, 256),
                                        batch_size=batch_size)

test_datagen = ImageDataGenerator(rescale=1./255)

test_gen = test_datagen.flow_from_directory('../data/augmentation_validation',
                                        target_size=(256, 256),
                                        batch_size=batch_size)

# Check if a pretrained model exists
model_path = "model_augmented.h5"
model_file = Path(model_path)
if model_file.is_file():
    model = load_model(model_path)
else: # Start from scratch

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

    learning_rate = 5e-4
    decay = learning_rate/100
    optimizer = Adam(lr=learning_rate, decay=decay)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#compile and fit model

earlystop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=15, \
                          verbose=0, mode='auto')
callbacks_list = [earlystop]

n_epochs = 50
history_dict = None
# Save model and history dict between every epoch
for i in range(n_epochs):
    print("\nEpoch " + str(i+1) + "/" + str(n_epochs))
    history = model.fit_generator(generator=train_gen, validation_data=test_gen,
                        steps_per_epoch=train_steps, validation_steps=validation_steps,
                        epochs=1, verbose=2, callbacks=callbacks_list)
    history_dict = add_to_history(history_dict, history)
    model.save('model_augmented.h5')
    with open('history_augmented', 'wb') as file_pi:
        pickle.dump(history_dict, file_pi)



