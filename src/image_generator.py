import numpy as np
from skimage.io import imread
from skimage.transform import resize
from sklearn.feature_extraction.image import extract_patches_2d
from keras.utils import Sequence, to_categorical


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