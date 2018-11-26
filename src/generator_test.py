from image_generator import *
from keras.layers import Flatten, Dense
from keras.models import Sequential

img_shape = (256, 256)
batch_size = 32
train_gen, test_gen = get_generators(n_total=64, batch_size=batch_size, image_shape=img_shape)

# Define the most simple keras model
model = Sequential()
model.add(Flatten(input_shape=(img_shape[0], img_shape[1], 3)))
model.add(Dense(5, activation="softmax"))

# Compile and fit the model
model.compile(optimizer='adam', loss='categorical_crossentropy')

model.fit_generator(generator=train_gen,
                    steps_per_epoch=len(train_gen),
                    validation_data=test_gen,
                    epochs=2,
                    verbose=1)
