import numpy as np
import pickle

#import engine.turboprop as tp
from engine.nnTensor import Dense, Sequential, CrossEntropyLoss
# import matplotlib.pyplot as plt

# loading MNIST from bin files directly
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
                                   offset=16).reshape(n_train, 28 * 28)
# 2 int32 header (8 bytes): magic number 2049, nb_items=60_000
mnist_train_labels = np.frombuffer(train_labels_buf,
                                   dtype='B',
                                   count=n_train,
                                   offset=8)

X = mnist_train_images.reshape([-1, batch_size, 28 * 28]).astype('float32') / 255  # normalize
y = mnist_train_labels.reshape([-1, batch_size]).astype('int64') # just to pass asssert, shd clean later

# same for test set (n = 10_000)
mnist_test_images = np.frombuffer(test_images_buf,
                                  dtype='B',
                                  count=28 * 28 * n_test,
                                  offset=16).reshape(n_test, 28 * 28)
mnist_test_labels = np.frombuffer(test_labels_buf,
                                  dtype='B',
                                  count=n_test,
                                  offset=8)

X_test = mnist_test_images.reshape([-1, batch_size, 28 * 28]).astype('float32') / 255  # normalize
y_test = mnist_test_labels.reshape([-1, batch_size]).astype('int64') # just to pass asssert, shd clean late

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

model = Sequential([
    Dense(28 * 28, 32),
    Dense(32, 10, relu=False)
    ])

#print(model)

# Params
learning_rate = 1e-1

# Initialize the loss function
loss_fn = CrossEntropyLoss()

# Training

def train_loop(X, y, model, loss_fn):
    n_batch = y.shape[0]
    for batch in range(n_batch):
        model.zero_grad()
        # Compute prediction and loss
        z_pred = model(X[batch].T)
        loss = loss_fn(z_pred, y[batch].reshape(1, -1))

        # Backpropagation
        loss.backward()
        for p in model.parameters():
            p.array -= learning_rate * p.grad

        if batch == n_batch - 1:
            loss = loss.value.squeeze()
            print(f'Batch: {batch} - loss: {loss:>7f}')


def test_loop(X_test, y_test, model, loss_fn):
    # need a way to set the model in eval mode (stop gradient calc)
    n_batch = y_test.shape[0]
    test_loss, correct = 0, 0
    for batch in range(n_batch):
        z_test_pred = model(X_test[batch].T)
        test_loss += loss_fn(z_test_pred, y_test[batch].reshape(1, -1)).value.squeeze()
        
        correct += (z_test_pred.array.argmax(axis=0) == y_test[batch]).sum()
    test_loss /= n_batch
    correct /= n_test
    print(f"Test Error: \nAccuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


# Training loop
#n_epoch = 300
n_epoch = 100
for epoch in range(n_epoch):
    print(f'Epoch #{epoch + 1}:')
    train_loop(X, y, model, loss_fn)
    test_loop(X_test, y_test, model, loss_fn)

# Writing the model to a file using pickle
fname_model = 'model_1H_32.turbo'
with open(fname_model, 'wb') as file:
    pickle.dump(model, file)
print('Done!')
