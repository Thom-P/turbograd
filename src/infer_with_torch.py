import torch
from torch import nn
from torchvision import transforms
from PIL import Image

import sys

with Image.open("test.png") as im:
    im_28_28 = im.resize((28, 28))
    X = transforms.ToTensor()(im_28_28) # this also does the 0 - 1 scaling

#print(X.dtype)
#print(X.shape)

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
y_pred = model(X)
#print(y_pred)
ind_max = y_pred.argmax()
print(f'Detected a {ind_max}')
#softmax = nn.Softmax(dim=1)
#pred_probab = softmax(y_pred)
#print(pred_probab)
#print(pred_probab.sum())
