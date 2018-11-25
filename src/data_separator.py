import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from numpy.core.defchararray import add, replace

# Separate the data set into test and train data

# Read filenames from a text file listing all the images
filenames = np.genfromtxt('../data/train_list.txt', dtype=str)

# Read in the labels of images
all_labels = np.genfromtxt('../data/trainLabels.csv', skip_header=1, dtype=str, delimiter=',')

# Plot the distribution of classes in the original data
classes, counts = np.unique(all_labels[:,1], return_counts=True)
plt.figure()
plt.bar(classes, counts)
plt.title('Distribution of retinopathy severity classes')
plt.xlabel('Class')
plt.ylabel('Count')
plt.savefig('../results/class_distribution.png')

# Save class distribution in original data
class_dist = np.asarray((classes, counts), dtype=np.int).T
np.savetxt(fname='../results/class_distribution.csv', X=class_dist, delimiter=',')

# Take a random sample of 3000 images aside as the test set
np.random.seed(38)
np.random.shuffle(all_labels)
test_labels = all_labels[:3000, :]
train_labels = all_labels[3000:, :]

# Plot and save distribution of test data
classes, counts = np.unique(test_labels[:,1], return_counts=True)
plt.figure()
plt.bar(classes, counts)
plt.title('Distribution of retinopathy severity classes in test data')
plt.xlabel('Class')
plt.ylabel('Count')
plt.savefig('../results/class_distribution_test.png')

class_dist = np.asarray((classes, counts), dtype=np.int).T
np.savetxt(fname='../results/class_distribution_test.csv', X=class_dist, delimiter=',')

# PLot and save distribution of train data
classes, counts = np.unique(train_labels[:,1], return_counts=True)
plt.figure()
plt.bar(classes, counts)
plt.title('Distribution of retinopathy severity classes in train data')
plt.xlabel('Class')
plt.ylabel('Count')
plt.savefig('../results/class_distribution_train.png')

class_dist = np.asarray((classes, counts), dtype=np.int).T
np.savetxt(fname='../results/class_distribution_train.csv', X=class_dist, delimiter=',')

# Save filenames separately
test_filenames = add(test_labels[:,0], np.full(shape=test_labels[:,0].shape, fill_value='.jpeg'))
np.savetxt(fname='../data/test_filenames.txt', X=test_filenames, delimiter='', fmt='%s')
train_filenames = add(train_labels[:,0], np.full(shape=train_labels[:,0].shape, fill_value='.jpeg'))
np.savetxt(fname='../data/train_filenames.txt', X=train_filenames, delimiter='', fmt='%s')