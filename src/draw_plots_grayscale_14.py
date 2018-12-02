import numpy as np
from performance_plots import *
import pickle
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

# PLot accuracy and loss
with open('history_grayscale_augmented_14', 'rb') as file_pi:
        hist_dict = pickle.load(file_pi)

plot_history(hist_dict, title='Loss and Accuracy', fname='../results/CNN_grayscale_augmented_14.png')


import itertools
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from pylab import rcParams

def plot_confusion_matrix_14(cm, classes, fname,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    https://www.kaggle.com/grfiv4/plot-a-confusion-matrix
    """
    if normalize:
        cm2 = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    else:
        cm2 = cm

    plt.figure(figsize=(8,7))
    plt.imshow(cm2, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    fmt = '.1f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd')  + '\n(' + format(cm2[i, j]*100, fmt) + ')',
                 horizontalalignment="center", verticalalignment='center',
                 color="white" if cm[i, j] > thresh else "black")

    plt.gcf().subplots_adjust(bottom=0.15, left=0.85)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(fname)


def save_confusion_matrix_14(y_true, y_pred, result_path='../results/'):
    '''
    Saves confusion matrices (raw and normalized) as .csv and plots of them as png
    :param y_true: 1D array of true class labels
    :param y_pred: 1D array of predicted class labels
    :param result_path: path to save confusion matrices and the plots
    :return: None
    '''
    # Save confusion matrix and normalized confusion matrix:
    cm = confusion_matrix(y_true, y_pred)
    np.savetxt(fname=result_path+"confusion_matrix_raw.csv", X=cm, delimiter=",")
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    np.savetxt(fname=result_path+"confusion_matrix_norm.csv", X=cm_norm, delimiter=",")
    # Plot the confusion matrices
    classes = np.arange(2)
    classes[1] = 4
    plot_confusion_matrix_14(cm, classes=classes, fname=result_path+"confusion_matrix_norm.png",
                          normalize=True)


# plot confusion matrix
batch_size = 1
img_shape = (256, 256)

#datagen = ImageDataGenerator(
#        rotation_range=40,
#        shear_range=0.2,
#        zoom_range=0.2,
#        rescale=1./255,
#        horizontal_flip=True,
#        fill_mode='nearest')

#rawgen = ImageDataGenerator(
#	rotation_range=40,
#	shear_range=0.2,
#	zoom_range=0.2,
#	rescale=1./255,
#	horizontal_flip=True,
#	fill_mode='nearest')

rawgen = ImageDataGenerator(rescale=1./255)

test_gen = rawgen.flow_from_directory('../data/grayscale_14_test',
                                      target_size=(256, 256),
                                      batch_size=batch_size,
                                      shuffle=False)

model = load_model('model_grayscale_augmented_14.h5')

y_pred = model.predict_generator(test_gen, steps=2272)

y_pred = np.argmax(y_pred, axis=1)
#print(y_pred[0:300])
print('Y_pred shape is:')
print(y_pred.shape)
print('\n')

y_true = test_gen.classes
#print(y_true[0:300])
print('Y_true shape is:')
print(y_true.shape)
print('\n')
save_confusion_matrix_14(y_true, y_pred, result_path='../results/CNN_grayscale_augmented_14_test_')
