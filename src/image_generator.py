import numpy as np
from skimage.io import imread
from skimage.transform import resize
from keras.utils import Sequence, to_categorical


class ImageGenerator(Sequence):

    def __init__(self, image_filenames, labels, batch_size, image_shape):
        self.image_filenames, self.labels = image_filenames, labels
        self.image_shape, self.batch_size = image_shape, batch_size

    def __len__(self):
        return int(np.ceil(len(self.image_filenames) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.image_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]

        return np.array([
            resize(imread(file_name), self.image_shape)
            for file_name in batch_x]), to_categorical(np.array(batch_y), num_classes=5)