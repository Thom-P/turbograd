import tkinter as tk
import tkinter.font as font
from PIL import Image, ImageTk, ImageDraw


class MyWindow:
    def __init__(self, root):
        root.title('Single Digit Recognition')
        self.image = Image.new('L', (400, 400), color=0)
        self.draw = ImageDraw.Draw(self.image)
        self.tkImage = ImageTk.PhotoImage(self.image)
        self.canvas = tk.Canvas(root, width=400, height=400)
        self.imageContainer = self.canvas.create_image(0, 0, anchor="nw",
                                                       image=self.tkImage)
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

    def detect(self):
        self.image.save('test.png')

    # def doneStroke(self, event):
    #   pass


root = tk.Tk()
gui = MyWindow(root)
root.mainloop()
