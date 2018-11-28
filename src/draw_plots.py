import numpy as np
from performance_plots import *
import pickle
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

# PLot accuracy and loss
with open('history_augmented', 'rb') as file_pi:
        hist_dict = pickle.load(file_pi)

plot_history(hist_dict, title='testing', fname='../results/CNN_augmented.png')

# plot confusion matrix
batch_size = 32
img_shape = (256, 256)

rawgen = ImageDataGenerator()
test_gen = rawgen.flow_from_directory('../data/augmentation_validation',
                                      target_size=(256, 256),
                                      batch_size=batch_size,
                                      shuffle=False)

model = load_model('model_downsampled.h5')

y_pred = model.predict_generator(test_gen, steps=1000)
y_pred = np.argmax(y_pred, axis=1)
print(y_pred.shape)

y_true = test_gen.labels.astype(int)
save_confusion_matrix(y_true, y_pred, result_path='../results/CNN_downsampled')
