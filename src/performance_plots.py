import numpy as np
import itertools
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from pylab import rcParams

def plot_confusion_matrix(cm, classes, fname,
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
    plot_confusion_matrix(cm, classes=np.arange(5), fname=result_path+"confusion_matrix_norm.png",
                          normalize=True)


# Fucntion to be used to plot statistics
def prepare_standardplot(title, xlabel):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle(title)
    ax1.set_ylabel('categorical cross entropy')
    ax1.set_xlabel(xlabel)
    ax1.set_yscale('log')
    ax2.set_ylabel('accuracy [% correct]')
    ax2.set_xlabel(xlabel)
    return fig, ax1, ax2

def finalize_standardplot(fig, ax1, ax2):
    ax1handles, ax1labels = ax1.get_legend_handles_labels()
    if len(ax1labels) > 0:
        ax1.legend(ax1handles, ax1labels)
    ax2handles, ax2labels = ax2.get_legend_handles_labels()
    if len(ax2labels) > 0:
        ax2.legend(ax2handles, ax2labels)
    fig.tight_layout()
    plt.subplots_adjust(top=0.9)

def plot_history(history, title, fname):
    fig, ax1, ax2 = prepare_standardplot(title, 'epoch')
    ax1.plot(history['loss'], label = "training")
    ax1.plot(history['val_loss'], label = "validation")
    ax2.plot(history['acc'], label = "training")
    ax2.plot(history['val_acc'], label = "validation")
    finalize_standardplot(fig, ax1, ax2)
    plt.savefig(fname)

def add_to_history(history_dict, history):
    dict_tmp = history.history
    # For all the fields, join the lists together
    if history_dict is None:
        history_dict = dict_tmp
    else:
        for k in history_dict.keys():
            history_dict[k] = history_dict[k] + dict_tmp[k]
    return history_dict

# For testing
if __name__ == "__main__":
    # Sample labels
    np.random.seed(38)
    y_true = np.random.choice(np.arange(5), size=1000)
    y_pred = np.random.choice(np.arange(5), size=1000)

    save_confusion_matrix(y_true, y_pred)

