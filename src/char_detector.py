import pickle

import numpy as np
from PIL import Image, ImageFilter, ImageOps
import tkinter as tk

from utils.gui import MyWindow
from utils.crop_array import crop_image_array

# Build dictionary for class index -> ascii conversion
indices_keys = list(range(47))
ascii_values = list(range(48, 58)) + list(range(65, 91)) \
    + [97, 98, 100, 101, 102, 103, 104, 110, 113, 114, 116]
ascii_from_class = dict(zip(indices_keys, ascii_values))

# Reading the trained model from file
with open('model_1h_128.tg', 'rb') as fmodel:
    model = pickle.load(fmodel)
# todo: switch to eval mode when feature added


# Preprocess image and detect char
def detect(image, msg):
    """The user drawn 400x400 input image is preprocessed using the following
    procedure (see EMNIST paper): resized to 128x128, blurred with narrow 
    gaussian filter to soften edges, extraction of square bounding box,
    addition of a two pixel border and final downsizing to 28x28 size.
    The flattened image is then forward-passed through the trained model to
    infer its corresponding class and associated char.
    """
    im128 = image.resize((128, 128))
    im128 = im128.filter(ImageFilter.GaussianBlur(radius=1))
    arr = np.array(im128)
    if not arr.any():  # empty image/array
        return

    # EMNIST images are transposed compared to MNIST
    im_crop = crop_image_array(arr).T
    im_norm = Image.fromarray(im_crop)
    # Add 2 pixel border
    im_norm = ImageOps.expand(im_norm, border=2, fill=0)
    # Downsample to 28x28 image
    im_28 = im_norm.resize((28, 28), resample=Image.Resampling.BICUBIC)
    # Normalize pixel values in 0-255 range
    im_28 = ImageOps.autocontrast(im_28)

    # Prepare input for Neural Net model
    x = np.array(im_28).reshape(28 * 28, 1).astype(np.float32) / 255.
    z_pred = model(x)
    ind_max = z_pred.array.argmax()
    char_pred = chr(ascii_from_class[ind_max])
    msg.set(f'Detected char: {char_pred}')
    return


root = tk.Tk()
win = MyWindow(root, detect)
root.mainloop()
