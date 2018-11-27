import numpy as np
from skimage import color, exposure
from skimage.io import imread

def change_exposure(fname):
    img = imread('../data/converted/'+ fname + '.jpeg')
    hsv = color.rgb2hsv(img)
    hsv[:, :, 2] = exposure.equalize_hist(hsv[:, :, 2])
    img = color.hsv2rgb(hsv)
    img = img / 255
    np.save('../data/arrays/' + fname + '.npy', img)
    
full_labels = np.genfromtxt('../data/trainLabels.csv', skip_header=1, dtype=str, delimiter=',')
filenames = full_labels[:,0] 

for fname in filenames:
    change_exposure(fname)
