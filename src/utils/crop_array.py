import numpy as np


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
