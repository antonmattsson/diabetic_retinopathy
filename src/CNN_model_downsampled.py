import numpy as np
from image_generator import *
from keras.layers import Flatten, Dense, ZeroPadding2D, Conv2D, Activation, MaxPooling2D, Dropout
from keras.layers import Flatten, Dense, Conv2D, Activation, MaxPooling2D, Dropout
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from performance_plots import *          
import pickle

# Set the number of training samples
n_total = 2400
batch_size = 32
img_shape = (256, 256)

# Construc generators for the training and validation data
# Downsample zero grade images, load from prepocessed numpy arrays
train_gen, test_gen = get_generators(n_total=n_total, batch_size=batch_size)

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
                        steps_per_epoch=len(train_gen),
                        epochs=1, verbose=2, callbacks=callbacks_list)
    history_dict = add_to_history(history_dict, history)
    model.save('model_downsampled.h5')
    with open('history_downsampled', 'wb') as file_pi:
        pickle.dump(history_dict, file_pi)



