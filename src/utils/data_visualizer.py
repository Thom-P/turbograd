"""Visualize random training set images from the 47 classes
"""
import numpy as np
from PIL import Image

from data_loader import load_data

# Load train dataset
n_train = 112_800  # number of images in train set
n_class = 47  # number of classes
train_images_f = '../../EMNIST_dataset/emnist-balanced-train-images-idx3-ubyte'
train_labels_f = '../../EMNIST_dataset/emnist-balanced-train-labels-idx1-ubyte'

train_images, train_labels = load_data(train_images_f, train_labels_f, n_train)

# Choose 10 random occurences of each of the 47 classes and put the
# corresponding 28 x 28 images in a 10 x 47 array
np.random.seed(42)
n_set = 10
im_array = np.empty((n_set * 28, n_class * 28), dtype=np.uint8)
for class_ in range(n_class):
    ind_set = (train_labels == class_).nonzero()[0]
    ind_select = np.random.choice(ind_set, size=n_set, replace=False)
    for i, ind in enumerate(ind_select):
        im_array[i * 28:(i + 1) * 28, class_ * 28:(class_ + 1) * 28] = \
            train_images[ind].reshape(28, 28).T

im = Image.fromarray(im_array)
im.show()
# im.save('train_set_samples.png')
