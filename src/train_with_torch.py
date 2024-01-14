import torch
from torch import nn
from torchvision import datasets, transforms
import sys
import numpy as np
import matplotlib.pyplot as plt
# mnist_raw_train = torchvision.datasets.MNIST(root='dataset', train=True, transform=np.asarray, download=True)
# mnist_raw_test = torchvision.datasets.MNIST(root='dataset', train=False, transform=np.asarray, download=True)

'''
# loading mnist using pytorch
mnist_raw_train = datasets.MNIST(root='../MNIST_dataset',
                                 train=True,
                                 transform=transforms.ToTensor(),
                                 download=True)
mnist_raw_test = datasets.MNIST(root='../MNIST_dataset',
                                train=False,
                                transform=transforms.ToTensor(),
                                download=True)
n_train = len(mnist_raw_train)  # 60_000
n_test = len(mnist_raw_test)  # 10_000
batch_size = 500  # use a divisor of n_train and n_test

print(mnist_raw_train.data.shape)
print(mnist_raw_train.data[0])
X = mnist_raw_train.data.reshape([-1, batch_size, 28, 28]) / 255  # normalize
y = mnist_raw_train.targets.reshape([-1, batch_size])
print(y.shape)
print(y.dtype)

X_test = mnist_raw_test.data.reshape([-1, batch_size, 28, 28]) / 255  # normalize
y_test = mnist_raw_test.targets.reshape([-1, batch_size])
'''

# loading MNIST fron bin files directly
with open('../MNIST_dataset/MNIST/raw/train-images-idx3-ubyte', 'rb') as f:
    train_images_buf = f.read()
with open('../MNIST_dataset/MNIST/raw/train-labels-idx1-ubyte', 'rb') as f:
    train_labels_buf = f.read()
with open('../MNIST_dataset/MNIST/raw/t10k-images-idx3-ubyte', 'rb') as f:
    test_images_buf = f.read()
with open('../MNIST_dataset/MNIST/raw/t10k-labels-idx1-ubyte', 'rb') as f:
    test_labels_buf = f.read()

#  header = np.frombuffer(train_images_buf, dtype='>i4', count=4, offset=0)
#  could put asserts here

n_train = 60_000
n_test = 10_000
batch_size = 500  # use a divisor of n_train and n_test

# 4 int32 header (16 bytes): magic num=2051, n_image=60_000, n_row=28, n_col=28
mnist_train_images = np.frombuffer(train_images_buf,
                                   dtype='B',
                                   count=28 * 28 * n_train,
                                   offset=16).reshape(n_train, 28, 28)
# 2 int32 header (8 bytes): magic number 2049, nb_items=60_000
mnist_train_labels = np.frombuffer(train_labels_buf,
                                   dtype='B',
                                   count=n_train,
                                   offset=8)

X = mnist_train_images.reshape([-1, batch_size, 28, 28]).astype('float32') / 255  # normalize
y = mnist_train_labels.reshape([-1, batch_size])
X = torch.from_numpy(X.copy())
y = torch.from_numpy(y.copy())

# same for test set (n = 10_000)
mnist_test_images = np.frombuffer(test_images_buf,
                                  dtype='B',
                                  count=28 * 28 * n_test,
                                  offset=16).reshape(n_test, 28, 28)
mnist_test_labels = np.frombuffer(test_labels_buf,
                                  dtype='B',
                                  count=n_test,
                                  offset=8)

X_test = mnist_test_images.reshape([-1, batch_size, 28, 28]).astype('float32') / 255  # normalize
y_test = mnist_test_labels.reshape([-1, batch_size])
X_test = torch.from_numpy(X_test.copy())
y_test = torch.from_numpy(y_test.copy())

# print(X.shape)
# print(X.dtype)
# print(X_test.dtype)
# print(X_test.shape)
# print(y.shape)
# print(y.dtype)
# print(y_test.shape)
# print(y_test.dtype)


# plt.imshow(X[0, 0])
# plt.show()


# ***************** Define model

device = 'cpu'
print(f"Using {device} device")


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


model = NeuralNetwork().to(device)
print(model)

# Params
learning_rate = 1e-1

# Initialize the loss function
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) 

# Training
# Set the model to training mode - important for batch normalization
# and dropout layers
# Unnecessary in this situation but added for best practices


def train_loop(X, y, model, loss_fn, optimizer):
    n_batch = y.shape[0]
    model.train()
    for batch in range(n_batch):
        # Compute prediction and loss
        y_pred = model(X[batch])
        loss = loss_fn(y_pred, y[batch])

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch == n_batch - 1:
            loss = loss.item()
            print(f'Batch: {batch} - loss: {loss:>7f}')


def test_loop(X_test, y_test, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    n_batch = y_test.shape[0]

    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for batch in range(n_batch):
            y_test_pred = model(X_test[batch])
            test_loss += loss_fn(y_test_pred, y_test[batch]).item()
            correct += (y_test_pred.argmax(1) == y_test[batch]).type(torch.float).sum().item()
    test_loss /= n_batch
    correct /= n_test
    print(f"Test Error: \nAccuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


# Training loop
#n_epoch = 80
n_epoch = 80
for epoch in range(n_epoch):
    print(f'Epoch #{epoch + 1}:')
    train_loop(X, y, model, loss_fn, optimizer)
    test_loop(X_test, y_test, model, loss_fn)

print('Done!')
torch.save(model, 'model2.pth')

# torch.save(model.state_dict(), 'model_weights.pth')
