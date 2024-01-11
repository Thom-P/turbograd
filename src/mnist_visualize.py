import tkinter as tk
import tkinter.font as font
from PIL import Image, ImageTk, ImageDraw
import random
random.seed(42)
## tmp add detection here before packaging

import torch
from torch import nn
from torchvision import transforms, datasets

mnist_raw_test = datasets.MNIST(root='../MNIST_dataset',
                                train=False,
                                #transform=transforms.ToTensor(),
                                download=True)


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


model = torch.load('model.pth')
model.eval()

## GUI

class MyWindow:
    def __init__(self, root):
        root.title('Single Digit Recognition')
        self.image = Image.new('L', (400, 400), color=0)
        #self.draw = ImageDraw.Draw(self.image)
        self.tkImage = ImageTk.PhotoImage(self.image)
        self.canvas = tk.Canvas(root, width=400, height=400)
        self.imageContainer = self.canvas.create_image(0, 0, anchor="nw",
                                                       image=self.tkImage)
        self.digit = tk.StringVar()
        self.digit.set('Detected digit: ?')
        self.label = tk.Label(root, textvariable=self.digit) #, relief=RAISED )
        self.label['font'] = font.Font(size=30) 
        
        self.line = []
        #self.canvas.bind("<Button-1>", self.initLine)
        #self.canvas.bind("<B1-Motion>", self.drawLine)
        # self.canvas.bind("<B1-ButtonRelease>", self.doneStroke)

        self.sampleButton = tk.Button(root, text="Sample",
                                     command=self.sampleImage)
        self.sampleButton['font'] = font.Font(size=30)

        self.detectButton = tk.Button(root, text="Detect",
                                      command=self.detect)
        self.detectButton['font'] = font.Font(size=30)

        self.label.pack()
        self.canvas.pack()
        self.sampleButton.pack()  # side='left'
        self.detectButton.pack()

    #def initLine(self, event):
    #    self.line = [(self.canvas.canvasx(event.x),
    #                  self.canvas.canvasy(event.y))]

    '''
    def drawLine(self, event):
        x, y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        self.line.append((x, y))
        self.draw.line(self.line, fill='white', width=15, joint='curve')
        self.tkImage = ImageTk.PhotoImage(self.image)
        self.canvas.itemconfig(self.imageContainer, image=self.tkImage)
        # self.canvas.create_line((self.line[-1][0], self.line[-1][1], x, y),
        #                         fill='white', width=10, capstyle='round')
    '''
    
    def sampleImage(self):
        ind = random.randint(0, 10000 - 1)
        raw_image = mnist_raw_test[ind][0]
        self.image = raw_image.resize((400, 400))
        #self.image = Image.new('L', (400, 400), color=0)
        #self.draw = ImageDraw.Draw(self.image)
        self.tkImage = ImageTk.PhotoImage(self.image)
        self.canvas.itemconfig(self.imageContainer, image=self.tkImage)
        self.digit.set('Detected digit: ?')  

    def detect(self):
        #self.image.save('test.png')
        im_28_28 = self.image.resize((28, 28))
        X = transforms.ToTensor()(im_28_28) # this also does the 0 - 1 scaling
        y_pred = model(X)
        #print(y_pred)
        ind_max = y_pred.argmax()
        #print(f'Detected a {ind_max}')
        self.digit.set(f'Detected digit: {ind_max}')
    # def doneStroke(self, event):
    #   pass


root = tk.Tk()
gui = MyWindow(root)
root.mainloop()
