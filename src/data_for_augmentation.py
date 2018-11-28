import numpy as np
from skimage.io import imread, imsave
from skimage import color, exposure
from numpy.core.defchararray import add, replace

def change_exposure(fname, label, folder):
    img = imread('../data/converted/'+ fname + '.jpeg')
    hsv = color.rgb2hsv(img)
    hsv[:, :, 2] = exposure.equalize_hist(hsv[:, :, 2])
    img = color.hsv2rgb(hsv)
    img = img / 255
    imsave(folder + label + '/' + fname + '.png', img)

full_filenames = np.genfromtxt('../data/train_filenames.txt', dtype=str)

# Read the labels file
full_labels = np.genfromtxt('../data/trainLabels.csv', skip_header=1, dtype=str, delimiter=',')
# Keep only labels of data that can be used in training
full_samples = replace(full_filenames, ".jpeg", "")
full_mask = np.isin(full_labels[:, 0], full_samples)
trainable_labels = np.copy(full_labels[full_mask, :])

# Downsample the zero grade, keeping only the first 5000
# Randomize order
np.random.seed(1234)
np.random.shuffle(trainable_labels)
# Arrange by a stable sort (mergesort)
trainable_labels = np.copy(trainable_labels[trainable_labels[:,1].argsort(kind='mergesort')])
# Remove extra zeros
zeros_left = 5000
if zeros_left > 0:
    _, counts = np.unique(trainable_labels[:,1], return_counts=True)
    n_zeros = counts[0]
    downsampled_labels = np.copy(trainable_labels[(n_zeros-zeros_left):, :])
else:
    downsampled_labels = np.copy(trainable_labels)

# Use downsampled set for training
n_train = downsampled_labels.shape[0]
np.random.shuffle(downsampled_labels)
train_labels = np.copy(downsampled_labels)

# Exclude training samples from the original data and use the rest for testing
np.random.shuffle(trainable_labels)
exclusion = np.isin(trainable_labels[:, 0], train_labels[:, 0], invert=True)
validation_labels = np.copy(trainable_labels[exclusion, :])
n_validation = validation_labels.shape[0]

print(n_train, n_validation)

# Save images into subfolders by class
for i in range(n_train):
    change_exposure(fname=train_labels[i, 0],
                    label=train_labels[i, 1],
                    folder='../data/augmentation_train/')

for j in range(n_validation):
    change_exposure(fname=validation_labels[j, 0],
                    label=validation_labels[j, 1],
                    folder='../data/augmentation_validation/')
