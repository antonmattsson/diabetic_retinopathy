import numpy as np
from skimage.io import imread
from skimage.transform import resize
from sklearn.feature_extraction.image import extract_patches_2d
from keras.utils import Sequence, to_categorical
from numpy.core.defchararray import add, replace
import matplotlib.pyplot as plt

class ImageGenerator(Sequence):
    '''
    Class for generating image batches from the image files
    :param image_filenames: 1D numpy array (or list) of file names of the images
    :param labels: 1D numpy array with the labels corresponding to each image
    :param batch_size: integer giving the batch size to be used in training the network
    :param image_shape: tuple of two integers. All images will be compressed to this shape
    '''
    def __init__(self, image_filenames, labels, batch_size, image_shape):
        self.image_filenames, self.labels = image_filenames, labels
        self.image_shape, self.batch_size = image_shape, batch_size

    def __len__(self):
        return int(np.ceil(len(self.image_filenames) / float(self.batch_size)))

    # Helper function to read and preprocess images
    def _read_image(self, filename):
        image = resize(imread(filename), self.image_shape)
        # Normalize pixel values between 0 and 1
        image = image / 255
        return image


    def __getitem__(self, idx):
        batch_x = self.image_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]

        return np.array([self._read_image(file_name) for file_name in batch_x]),\
            to_categorical(np.array(batch_y), num_classes=5)


class PatchGenerator(ImageGenerator):

    def __init__(self, image_filenames, labels, batch_size, patch_shape, n_patches):
        self.image_filenames, self.labels = image_filenames, labels
        self.batch_size = batch_size
        self.patch_shape, self.n_patches = patch_shape, n_patches

    def _read_image(self, filename):
        image = imread(filename)
        # Normalize pixel values between 0 and 1
        image = image / 255
        patches = extract_patches_2d(image, patch_size=self.patch_shape,
                                     max_patches=self.n_patches, random_state=38)
        return patches


class ArrayGenerator(Sequence):
    '''
    Class for generating arrays for training
    :param filenames: 1D array of filenames to read from, ending with .npy
    :param labels: 1D array of strings, labels
    :param batch_size: integer giving the batch size to be used in training the network
    '''
    def __init__(self, filenames, labels, batch_size):
        self.filenames = filenames
        self.labels, self.batch_size = labels, batch_size

    def __len__(self):
        return int(np.ceil(len(self.filenames) / float(self.batch_size)))

    def read_array(self, filename):
        array = np.load(filename)
        return array

    def __getitem__(self, idx):
        batch_x = self.filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]

        return np.array([np.load(file_name) for file_name in batch_x]),\
            to_categorical(np.array(batch_y), num_classes=5)


def get_generators(n_total, batch_size, image_shape=None, type='array', zeros_left=5000):
    '''
    Construct generators for training and validation data
    Zero grade images are downsampled
    :param n_total: number of total images to use (training plus validation)
    :param batch_size: batch size used in training
    :param image_shape: image size used in training
    :param zeros_left: how many images of grade zero should be left in the pool
                       use a negative value to keep all the zeros
    :return: train_gen: generator of training data
             test_gen: generator of validation data
    '''
    # Set the number of training samples
    n_train = int(np.ceil(n_total * 0.8))
    n_test = int(np.floor(n_total * 0.2))

    # Read filenames from a text file listing all the images
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
    if zeros_left > 0:
        _, counts = np.unique(trainable_labels[:,1], return_counts=True)
        n_zeros = counts[0]
        downsampled_labels = np.copy(trainable_labels[(n_zeros-zeros_left):, :])
    else:
        downsampled_labels = np.copy(trainable_labels)

    # Randomize and choose training data
    np.random.shuffle(downsampled_labels)
    train_labels = downsampled_labels[:n_train, :]
    #test_labels = downsampled_labels[n_train:(n_train + n_test)]
    # Exclude training samples from the original data and choose test data among them
    np.random.shuffle(trainable_labels)
    exclusion = np.isin(trainable_labels[:, 0], train_labels[:, 0], invert=True)
    valid_labels = np.copy(trainable_labels[exclusion, :])
    test_labels = np.copy(valid_labels[:n_test, :])

    # Print the counts of each class in test and train data
    _, train_counts = np.unique(train_labels[:, 1], return_counts=True)
    print("\nTrain distribution:")
    print(train_counts/np.sum(train_counts))
    _, test_counts = np.unique(test_labels[:, 1], return_counts=True)
    print("\nTest distribution:")
    print(test_counts/np.sum(test_counts))
    print("\n")

    if type == 'array':
        # Add .npy file ending
        train_filenames = add(train_labels[:, 0], np.full(shape=n_train, fill_value='.npy'))
        test_filenames = add(test_labels[:, 0], np.full(shape=n_test, fill_value='.npy'))
        # Add path of the data folder to the files
        train_filepaths = add(np.full(shape=train_filenames.shape, fill_value='../data/arrays/'), train_filenames)
        test_filepaths = add(np.full(shape=test_filenames.shape, fill_value='../data/arrays/'), test_filenames)

        # Create an instance of the image generator
        train_gen = ArrayGenerator(train_filepaths, train_labels[:, 1], batch_size)
        test_gen = ArrayGenerator(test_filepaths, test_labels[:, 1], batch_size)

    elif type == 'image':
        if image_shape is None:
            raise ValueError
        # Add .jpeg file ending
        train_filenames = add(train_labels[:, 0], np.full(shape=n_train, fill_value='.jpeg'))
        test_filenames = add(test_labels[:, 0], np.full(shape=n_test, fill_value='.jpeg'))
        # Add path of the data folder to the files
        train_filepaths = add(np.full(shape=train_filenames.shape, fill_value='../data/train/'), train_filenames)
        test_filepaths = add(np.full(shape=test_filenames.shape, fill_value='../data/train/'), test_filenames)

        # Create an instance of the image generator
        train_gen = ImageGenerator(train_filepaths, train_labels[:, 1], batch_size, image_shape)
        test_gen = ImageGenerator(test_filepaths, test_labels[:, 1], batch_size, image_shape)

    return train_gen, test_gen

if __name__ == "__main__":
    train_gen, test_gen = get_generators(n_total=2401, batch_size=1, image_shape=(512, 512), type='array')
    print((len(train_gen), len(test_gen)))
    #img = train_gen[0][0][0, ::]
    #print(img.shape)
    #plt.figure()
    #plt.imshow(img*255)
    #plt.show()