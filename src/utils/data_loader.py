import numpy as np


def load_data(image_fname, label_fname, n_image):
    """Load MNIST or EMNIST original format binary data: 28x28 images
    and associated class labels.
    """
    with open(image_fname, 'rb') as f:
        image_buf = f.read()
    with open(label_fname, 'rb') as f:
        label_buf = f.read()

    # 4 int32 header (16 bytes): magic num=2051, n_image, n_row=28, n_col=28
    images = np.frombuffer(image_buf,
                           dtype='B',
                           count=28 * 28 * n_image,
                           offset=16).reshape(n_image, 28 * 28)

    # 2 int32 header (8 bytes): magic number 2049, n_item
    labels = np.frombuffer(label_buf,
                           dtype='B',
                           count=n_image,
                           offset=8)
    return images, labels
