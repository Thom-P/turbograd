import torch
from torch import nn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
#mnist_raw_train = torchvision.datasets.MNIST(root='dataset', train=True, transform=np.asarray, download=True)
#mnist_raw_test = torchvision.datasets.MNIST(root='dataset', train=False, transform=np.asarray, download=True)

mnist_raw_train = datasets.MNIST(root='../MNIST_dataset', train=True, transform=transforms.ToTensor(), download=True)
mnist_raw_test = datasets.MNIST(root='../MNIST_dataset', train=False, transform=transforms.ToTensor(), download=True)

X_raw = mnist_raw_train.data
# normalize
X = X_raw / 255
print(X_raw.shape)
print(X_raw.dtype)
im_arr = X[50].numpy()
plt.imshow(im_arr)


y = mnist_raw_train.targets
print(y.shape)
print(y.dtype)

X_test_raw = mnist_raw_test.data
y_test = mnist_raw_test.targets
# normalize
X_test = X_test_raw / 255

from torch import nn
# Define model

device = 'cpu'
print(f"Using {device} device")

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)

#params
learning_rate = 1e-1
#batch_size = 64 # try all data

# Initialize the loss function
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) 

#train
# Set the model to training mode - important for batch normalization and dropout layers
# Unnecessary in this situation but added for best practices
model.train()
n_epoch = 1000
for epoch in range(n_epoch):
    # Compute prediction and loss
    y_pred = model(X)
    #print(y_pred.shape)
    #print(y_pred.dtype)
    
    loss = loss_fn(y_pred, y)

    # Backpropagation
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    if epoch % 10 == 0:
        loss = loss.item()
        print(f'Epoch: {epoch} - loss: {loss:>7f}')


# test

# Set the model to evaluation mode - important for batch normalization and dropout layers
# Unnecessary in this situation but added for best practices
model.eval()
size = len(y_test)
print(size)

test_loss, correct = 0, 0

# Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
# also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
with torch.no_grad():
    y_test_pred = model(X_test)
    test_loss = loss_fn(y_test_pred, y_test).item()
    correct = (y_test_pred.argmax(1) == y_test).type(torch.float).sum().item()

correct /= size
print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
