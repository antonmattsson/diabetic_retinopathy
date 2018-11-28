import numpy as np
from keras.layers import Flatten, Dense, ZeroPadding2D, Conv2D, Activation, MaxPooling2D, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Flatten, Dense, Conv2D, Activation, MaxPooling2D, Dropout
from keras.models import Sequential
from keras.callbacks import EarlyStopping
import pickle
from performance_plots import *

# Set the number of training samples
n_total = 2400
batch_size = 32
img_shape = (256, 256)

# Set data augmentation protocol
datagen = ImageDataGenerator(
        rotation_range=40,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

# Construct generators for the training and validation data using data augmentation
train_gen = datagen.flow_from_directory('../data/augmentation_train',
                                        target_size=(256, 256),
                                        batch_size=batch_size)

test_gen = datagen.flow_from_directory('../data/augmentation_test',
                                        target_size=(256, 256),
                                        batch_size=batch_size)

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

earlystop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=15, \
                          verbose=0, mode='auto')
callbacks_list = [earlystop]

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

n_epochs = 20
history_dict = None
# Save model and history dict between every epoch
for i in range(n_epochs):
    print("\nEpoch " + str(i+1) + "/" + str(n_epochs))
    history = model.fit_generator(generator=train_gen, validation_data=test_gen,
                        steps_per_epoch=60, validation_steps=10,
                        epochs=1, verbose=2, callbacks=callbacks_list)
    history_dict = add_to_history(history_dict, history)
    model.save('model_augmented.h5')
    with open('history_augmented', 'wb') as file_pi:
        pickle.dump(history_dict, file_pi)



