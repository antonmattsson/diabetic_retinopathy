import operator
from PIL import Image, ImageEnhance
from functools import reduce
from skimage.exposure import equalize_adapthist, rescale_intensity
import numpy as np
import cv2
from scipy import ndimage
import numpy as np
from skimage.io import imread
from datetime import datetime, timedelta

def preprocess(fname):

	# read image and convert to grayscale
	img = Image.open('../data/converted/' + fname + '.jpeg').convert('L')

	# remove salt and pepper noise with median filter (3x3 kernel)
	img = ndimage.median_filter(img, 3)

	# convert to numpy array
	im2arr = np.array(img)

	# apply Contrast-Limited Adaptive Histogram Equalisation
	img = equalize_adapthist(im2arr, clip_limit=0.02)
	img = rescale_intensity(img, out_range=(0, 255))

	# convert back to image
	# arr2im = Image.fromarray(img)
	# img = arr2im.convert('RGB')
	
	# normalize
	# img = img / 255
	
	# convert back to image
	img = Image.fromarray(img)
	img = img.convert('RGB')
	
	#np.save('../data/grayscale/' + fname + '.npy', img)
	img.save('../data/grayscale/' + fname + '.jpeg')


full_labels = np.genfromtxt('../data/trainLabels.csv', skip_header=1, dtype=str, delimiter=',')
filenames = full_labels[:,0]

now = datetime.now()
print(now.strftime("%Y-%m-%d %H:%M:%S.%f").rstrip('0'))
for fname in filenames:
	preprocess(fname)

now = datetime.now()
print(now.strftime("%Y-%m-%d %H:%M:%S.%f").rstrip('0'))

