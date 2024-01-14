import tkinter as tk
import tkinter.font as font
from PIL import Image, ImageTk, ImageDraw
import numpy as np

# TMP add detection here before packaging

import torch
from torch import nn
# from torchvision import transforms


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


model = torch.load('model2.pth')
model.eval()


# Input preprocess following MNIST procedure
def preprocess_image_arr(im_arr):
    is_row_empty = im_arr.any(1)
    row_start = np.argmax(is_row_empty)
    row_end = len(is_row_empty) - 1 - np.argmax(is_row_empty[::-1])

    is_col_empty = im_arr.any(0)
    col_start = np.argmax(is_col_empty)
    col_end = len(is_col_empty) - 1 - np.argmax(is_col_empty[::-1])

    # Crop/normalize image into square bounding box
    if row_end - row_start > col_end - col_start:
        n_pix = row_end - row_start + 1
        n_col = col_end - col_start + 1
        arr_norm = np.zeros((n_pix, n_pix), dtype='uint8')
        arr_norm[:, (n_pix - n_col) // 2:(n_pix - n_col) // 2 + n_col] \
            = im_arr[row_start:row_end + 1, col_start:col_end + 1]
    else:
        n_pix = col_end - col_start + 1
        n_row = row_end - row_start + 1
        arr_norm = np.zeros((n_pix, n_pix), dtype='uint8')
        arr_norm[(n_pix - n_row) // 2:(n_pix - n_row) // 2 + n_row, :] \
            = im_arr[row_start:row_end + 1, col_start:col_end + 1]

    im_norm = Image.fromarray(arr_norm, mode='L')
    im_20 = im_norm.resize((20, 20))  # downsample to 20x20
    arr_20 = np.array(im_20)

    # Compute center of mass (cm) of pixels
    # with correction of inverted axes bug and round instead of floor!
    y_cm, x_cm = (np.mgrid[0:20, 0:20] * arr_20).sum(1).sum(1) / arr_20.sum()
    row_cm, col_cm = round(y_cm), round(x_cm)

    # tmp arr of double size to include 20x20 centered on cm
    arr_40 = np.zeros((40, 40), dtype='uint8')
    # cm is at ind [20, 20] in arr40
    arr_40[20 - row_cm:20 - row_cm + 20, 20 - col_cm:20 - col_cm + 20] = arr_20

    # im_28 = Image.fromarray(arr_40[(40 - 28) // 2:(40 - 28) // 2 + 28, \
    #    (40 - 28) // 2:(40 - 28) // 2 + 28], mode='L')

    # cm at ind 14 in arr_28
    i_start = (40 - 28) // 2
    arr_28 = arr_40[i_start:i_start + 28, i_start:i_start + 28]

    return arr_28


# GUI
class MyWindow:
    def __init__(self, root):
        root.title('Single Digit Recognition')
        self.image = Image.new('L', (400, 400), color=0)
        self.draw = ImageDraw.Draw(self.image)
        self.tkImage = ImageTk.PhotoImage(self.image)
        self.canvas = tk.Canvas(root, width=400, height=400)
        self.imageContainer = self.canvas.create_image(0, 0, anchor="nw",
                                                       image=self.tkImage)
        self.digit = tk.StringVar()
        self.digit.set('Detected digit: ?')
        self.label = tk.Label(root, textvariable=self.digit)
        self.label['font'] = font.Font(size=30)

        self.line = []
        self.canvas.bind("<Button-1>", self.initLine)
        self.canvas.bind("<B1-Motion>", self.drawLine)
        # self.canvas.bind("<B1-ButtonRelease>", self.doneStroke)

        self.clearButton = tk.Button(root, text="Clear",
                                     command=self.clearCanvas)
        self.clearButton['font'] = font.Font(size=30)

        self.detectButton = tk.Button(root, text="Detect",
                                      command=self.detect)
        self.detectButton['font'] = font.Font(size=30)

        self.label.pack()
        self.canvas.pack()
        self.clearButton.pack()  # side='left'
        self.detectButton.pack()

    def initLine(self, event):
        self.line = [(self.canvas.canvasx(event.x),
                      self.canvas.canvasy(event.y))]

    def drawLine(self, event):
        x, y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        self.line.append((x, y))
        self.draw.line(self.line, fill='white', width=15, joint='curve')
        self.tkImage = ImageTk.PhotoImage(self.image)
        self.canvas.itemconfig(self.imageContainer, image=self.tkImage)
        # self.canvas.create_line((self.line[-1][0], self.line[-1][1], x, y),
        #                         fill='white', width=10, capstyle='round')

    def clearCanvas(self):
        self.image = Image.new('L', (400, 400), color=0)
        self.draw = ImageDraw.Draw(self.image)
        self.tkImage = ImageTk.PhotoImage(self.image)
        self.canvas.itemconfig(self.imageContainer, image=self.tkImage)
        self.digit.set('Detected digit: ?')

    def detect(self):
        arr = np.array(self.image)
        if not arr.any():  # empty image/array
            return
        arr_28x28 = preprocess_image_arr(arr)
        X = torch.from_numpy(arr_28x28) / 255
        X = X[None, :]  # add axis

        # this also does the 0 - 1 scaling
        # X = transforms.ToTensor()(im_28_28)

        y_pred = model(X)
        # print(y_pred)
        ind_max = y_pred.argmax()
        self.digit.set(f'Detected digit: {ind_max}')

    # def doneStroke(self, event):
    #   pass


root = tk.Tk()
gui = MyWindow(root)
root.mainloop()
