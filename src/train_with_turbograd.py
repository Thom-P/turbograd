import pickle

from turbograd.layers import Dense, Sequential, CrossEntropyLoss
from data_loader import load_data

n_train = 60_000
n_test = 10_000
batch_size = 500  # use a divisor of n_train and n_test

train_images_file = '../MNIST_dataset/MNIST/raw/train-images-idx3-ubyte'
train_labels_file = '../MNIST_dataset/MNIST/raw/train-labels-idx1-ubyte'

test_images_file = '../MNIST_dataset/MNIST/raw/t10k-images-idx3-ubyte'
test_labels_file = '../MNIST_dataset/MNIST/raw/t10k-labels-idx1-ubyte'

train_images, train_labels = load_data(train_images_file,
                                       train_labels_file, n_train)

test_images, test_labels = load_data(test_images_file,
                                     test_labels_file, n_test)

# Reshape for mini-batch processing and normalize pixel values between 0 - 1
X = train_images.reshape([-1, batch_size, 28 * 28]).astype('float32') / 255.
y = train_labels.reshape([-1, batch_size]).astype('int32')

# Same for test set
X_test = test_images.reshape([-1, batch_size,
                              28 * 28]).astype('float32') / 255.
y_test = test_labels.reshape([-1, batch_size]).astype('int32')

# Define a 2 layer model with 32 hidden units
model = Sequential([
    Dense(28 * 28, 32),
    Dense(32, 10, relu=False)
    ])
# print(model)

# Params
learning_rate = 1e-1
loss_fn = CrossEntropyLoss()


def train_loop(X, y, model, loss_fn):
    """Training loop: for every batch of the training set, reset gradient
    values, forward pass through model, compute loss, backpropagate
    gradient, and update model pameters.
    """
    n_batch = y.shape[0]
    for batch in range(n_batch):
        # Reset gradient to zero
        model.zero_grad()

        # Compute output logits and loss
        z_pred = model(X[batch].T)
        loss = loss_fn(z_pred, y[batch].reshape(1, -1))

        # Backpropagation and parameter update
        loss.backward()
        for p in model.parameters():
            p.array -= learning_rate * p.grad

        if (batch + 1) % 20 == 0:
            loss = loss.value.item()
            print(f'Batch: {batch + 1} - loss: {loss:>7f}')


def test_loop(X_test, y_test, model, loss_fn):
    """Testing loop: for every batch of the test set, forward pass through
    model, accumulate loss and number of correct inferences. Then compute
    average loss and accuracy.
    """
    # need a way to set the model in eval mode (stop gradient calc)
    n_batch = y_test.shape[0]
    test_loss, correct = 0, 0
    for batch in range(n_batch):
        z_test_pred = model(X_test[batch].T)
        test_loss += loss_fn(z_test_pred,
                             y_test[batch].reshape(1, -1)).value.item()
        correct += (z_test_pred.array.argmax(axis=0) == y_test[batch]).sum()
    test_loss /= n_batch
    correct /= n_test
    print(f'Test Error: \nAccuracy: {(100 * correct):>0.1f}%,'
          f'Avg loss: {test_loss:>8f} \n')


# Training over n_epochs
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
