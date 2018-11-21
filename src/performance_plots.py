import numpy as np
import itertools
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def _plot_confusion_matrix(cm, classes, fname,
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


def save_confusion_matrix(y_true, y_pred, result_path='../results/'):
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
    _plot_confusion_matrix(cm, classes=np.arange(5), fname=result_path+"confusion_matrix_norm.png",
                          normalize=True)

def _save_line_plot(x, y, xlabel, ylabel, title, fname, ylim=None):
    '''
    Plot lines
    :param x: 1D array for the x axis
    :param y: Dictionary: keys = line labels, values = 1D arrays of line y-coordinates
    :param xlabel: label of x-axis
    :param ylabel: label of y-axis
    :param title: The plot title
    :param fname: filename to save to
    :param ylim: limits for the y-axis
    :return: None
    '''

    # Set up the plot
    f, ax = plt.subplots(figsize=(12, 8))
    ax.set_title(title)
    # Plot lines from the dictionary
    for k in y.keys():
        ax.plot(x, y[k], label=k)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # Set limit for y axis to be able to compare between trials
    if ylim is not None:
        ax.set_ylim(*ylim)
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.9))
    # Add grid
    ax.grid(color='#999999', alpha=0.5, axis='y')
    plt.savefig(fname)

def save_accuracy_plot(n_epochs, train_acc, validation_acc,  result_path='../results/'):
    '''
    Save plot of train and validation accuracy as a function of epochs of training
    :param epochs: integer, number of epochs
    :param train_acc: 1D array of accuracies on training data
    :param validation_acc: 1D array of accuracies on validation data
    :param result_path: path to the results folder (plus a prefix)
    :return: None
    '''
    epochs = np.arange(1, n_epochs+1)
    _save_line_plot(x=epochs, y={'Train': train_acc, 'Validation':validation_acc},
                    xlabel='Epochs', ylabel='Accuracy', ylim=(0,1),
                    title='Prediction accuracy as a function of training epochs',
                    fname=result_path+'accuracy.png')


def save_loss_plot(n_epochs, train_loss, validation_loss,  result_path='../results/'):
    '''
    Save plot of train and validation loss as a function of epochs of training
    :param epochs: integer, number of epochs
    :param train_acc: 1D array of losses on training data
    :param validation_acc: 1D array of losses on validation data
    :param result_path: path to the results folder (plus a prefix)
    :return: None
    '''
    epochs = np.arange(1, n_epochs+1)
    _save_line_plot(x=epochs, y={'Train': train_loss, 'Validation': validation_loss},
                    xlabel='Epochs', ylabel='Accuracy',
                    title='Loss as a function of training epochs',
                    fname=result_path + 'loss.png')


# For testing
if __name__ == "__main__":
    # Sample labels
    np.random.seed(38)
    y_true = np.random.choice(np.arange(5), size=1000)
    y_pred = np.random.choice(np.arange(5), size=1000)
    # Confusion matrix
    save_confusion_matrix(y_true, y_pred)
    # Sample accuracies
    train_accuracy = sorted(np.random.uniform(high=0.8, size=100)) + np.random.uniform(-0.05, 0.05, size=100)
    test_accuracy =sorted(np.random.uniform(high=0.65, size=100)) + np.random.uniform(-0.05, 0.05, size=100)
    # Accuracy plot
    save_accuracy_plot(n_epochs=100, train_acc=train_accuracy, validation_acc=test_accuracy)
    # Sample loss
    train_loss = sorted(np.random.uniform(0.5, 2, size=100)) + np.random.uniform(-0.05, 0.05, size=100)
    test_loss = sorted(np.random.uniform(0.7, 2, size=100)) + np.random.uniform(-0.05, 0.1, size=100)
    # PLot loss
    save_loss_plot(n_epochs=100, train_loss=train_loss, validation_loss=test_loss)
