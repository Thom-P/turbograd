import tkinter as tk
import tkinter.font as font
from PIL import Image, ImageDraw


class MyWindow:
    def __init__(self, root):
        root.title('Single Digit Recognition')
        self.canvasSize = (400, 400)
        self.canvas = tk.Canvas(root, 
                                width=self.canvasSize[1], 
                                height=self.canvasSize[0],
                                background='black')
        self.line = []
        self.canvas.bind("<Button-1>", self.initLine)
        self.canvas.bind("<B1-Motion>", self.drawLine)
        # self.canvas.bind("<B1-ButtonRelease>", doneStroke)

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
        self.line = [(self.canvas.canvasx(event.x), self.canvas.canvasy(event.y))]

    def drawLine(self, event):
        x, y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        self.canvas.create_line((self.line[-1][0], self.line[-1][1], x, y),
                                fill='white', width=10, capstyle='round')
        self.line.append((x, y))

    def clearCanvas(self):
        self.canvas.delete('all')

    def detect(self):
        # Cannot save canvas to image (workaround: draw with PIL) 
        im = Image.new('L', self.canvasSize, color=0)
        draw = ImageDraw.Draw(im)
        draw.line(self.line, fill='white', width=10, joint='curve')
        im.save('test.png')
        

    # def doneStroke(event):
    #    canvas.itemconfigure('currentline', width=1)


root = tk.Tk()
gui = MyWindow(root)
root.mainloop()
