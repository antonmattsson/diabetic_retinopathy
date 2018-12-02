import numpy as np
from performance_plots import *
import pickle
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

'''
# PLot accuracy and loss
with open('history_grayscale_augmented_fast', 'rb') as file_pi:
        hist_dict = pickle.load(file_pi)

plot_history(hist_dict, title='Loss and Accuracy', fname='../results/CNN_grayscale_augmented_fast.png')
'''
# plot confusion matrix
batch_size = 40
img_shape = (256, 256)

#rawgen = ImageDataGenerator(rescale=1./255)

rawgen = ImageDataGenerator(
        rotation_range=40,
        shear_range=0.2,
        zoom_range=0.2,
        rescale=1./255,
        horizontal_flip=True,
        fill_mode='nearest')

test_gen = rawgen.flow_from_directory('../data/grayscale_aug_test',
                                      target_size=(256, 256),
                                      batch_size=batch_size,
                                      shuffle=False)



model = load_model('model_grayscale_augmented_fast.h5')


y_pred = model.predict_generator(test_gen, steps=75) 
y_pred = np.argmax(y_pred, axis=1)

print(y_pred)
#y_pred = np.argmax(y_pred, axis=1)
#y_pred = np.median(y_preds, axis=1)

print(y_pred.shape)

y_true = test_gen.classes
save_confusion_matrix(y_true, y_pred, result_path='../results/CNN_grayscale_augmented_test_')

