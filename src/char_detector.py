import pickle

import numpy as np
from PIL import Image, ImageFilter, ImageOps
import tkinter as tk

from utils.gui import MyWindow

# Build dictionary for class index -> ascii conversion
indices_keys = list(range(47))
ascii_values = list(range(48, 58)) + list(range(65, 91)) \
    + [97, 98, 100, 101, 102, 103, 104, 110, 113, 114, 116]
ascii_from_class = dict(zip(indices_keys, ascii_values))

# Reading the trained model from file
with open('model_1h_128.tg', 'rb') as fmodel:
    model = pickle.load(fmodel)
# need eval mode


# Crop and center region of interest following EMNIST procedure
def crop_image_array(im_arr):
    # Detect first/last non-empty rows and columns
    is_row_empty = im_arr.any(1)
    row_start = np.argmax(is_row_empty)
    row_end = len(is_row_empty) - 1 - np.argmax(is_row_empty[::-1])

    is_col_empty = im_arr.any(0)
    col_start = np.argmax(is_col_empty)
    col_end = len(is_col_empty) - 1 - np.argmax(is_col_empty[::-1])

    # Put region of interest into square bounding box
    if row_end - row_start > col_end - col_start:
        n_pix = row_end - row_start + 1
        n_col = col_end - col_start + 1
        arr_crop = np.zeros((n_pix, n_pix), dtype='uint8')
        arr_crop[:, (n_pix - n_col) // 2:(n_pix - n_col) // 2 + n_col] \
            = im_arr[row_start:row_end + 1, col_start:col_end + 1]
    else:
        n_pix = col_end - col_start + 1
        n_row = row_end - row_start + 1
        arr_crop = np.zeros((n_pix, n_pix), dtype='uint8')
        arr_crop[(n_pix - n_row) // 2:(n_pix - n_row) // 2 + n_row, :] \
            = im_arr[row_start:row_end + 1, col_start:col_end + 1]

    return arr_crop


# Preprocess image following EMNIST procedure
# and detect the char using our 3 layer model
def detect(image, msg):
    im128 = image.resize((128, 128))
    im128 = im128.filter(ImageFilter.GaussianBlur(radius=1))
    arr = np.array(im128)
    if not arr.any():  # empty image/array
        return

    # EMNIST images are transposed compared to MNIST
    im_crop = crop_image_array(arr).T
    im_norm = Image.fromarray(im_crop, mode='L')
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
