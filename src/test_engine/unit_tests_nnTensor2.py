import engine.turboprop as tp
from engine.nnTensor import Dense, Sequential, CrossEntropyLoss
import numpy as np
import torch
torch.set_printoptions(precision=6)

## same with model
learning_rate = 1e-1
X_np = np.random.randn(28 * 28, 500).astype(np.float32)
y_np = np.random.randint(0, 10, (1, 500))

layer1 = Dense(28 * 28, 16, label='L1')
layer2 = Dense(16, 10, relu=False, label='L2')
model = Sequential([
    layer1,
    layer2
    ])
print([(p.label, p.array.sum(), p.grad.sum()) for p in model.parameters()])
Z2_tp_model = model(X_np)
print([(p.label, p.array.sum(), p.grad.sum()) for p in model.parameters()])
loss_tp_model = CrossEntropyLoss()(Z2_tp_model, y_np)
print(loss_tp_model.value)
print([(p.label, p.array.sum(), p.grad.sum()) for p in model.parameters()])
#print([p.label for p in model.parameters()])
loss_tp_model.backward()
print([(p.label, p.array.sum(), p.grad.sum()) for p in model.parameters()])
for p in model.parameters():
    p.array -= learning_rate * p.grad
print([(p.label, p.array.sum(), p.grad.sum()) for p in model.parameters()])
model.zero_grad()
print([(p.label, p.array.sum(), p.grad.sum()) for p in model.parameters()])


'''
w1_np = layer1.weights.array.copy()
b1_np = layer1.biases.array.copy()
w2_np = layer2.weights.array.copy()
b2_np = layer2.biases.array.copy()

W1_tp = tp.Tensor(w1_np)
b1_tp = tp.Tensor(b1_np)
Z1_tp = W1_tp @ X_np + b1_tp
A1_tp = Z1_tp.relu()
W2_tp = tp.Tensor(w2_np)
b2_tp = tp.Tensor(b2_np)
Z2_tp = W2_tp @ A1_tp + b2_tp

loss_tp = CrossEntropyLoss()(Z2_tp, y_np)
print(loss_tp)

##

W1 = torch.tensor(w1_np, requires_grad=True)
b1 = torch.tensor(b1_np, requires_grad=True) 
W2 = torch.tensor(w2_np, requires_grad=True)
b2 = torch.tensor(b2_np, requires_grad=True) 
X = torch.tensor(X_np)  
Z1 = W1 @ X + b1
Z1.retain_grad()
A1 = torch.nn.ReLU()(Z1)
A1.retain_grad()
Z2 = W2 @ A1 + b2
Z2.retain_grad()
y = torch.tensor(y_np.squeeze())
loss = torch.nn.CrossEntropyLoss()(Z2.T, y)
print(loss)

loss_tp.backward()
loss.backward()

# let's try with absolute tol
# because order of op can be different, sign error can be expected
tol = 1e-3
with torch.no_grad():
    z1_err = (Z1 - Z1_tp.array).abs()
    print(f'Are Z1 fwd different?: {torch.any(z1_err > tol)}')
    a1_err = (A1 - A1_tp.array).abs()
    print(f'Are A1 fwd different?: {torch.any(a1_err > tol)}')
    z2_err = (Z2 - Z2_tp.array).abs()
    print(f'Are Z2 fwd different?: {torch.any(z2_err > tol)}')
    
    z1grad_err = (Z1.grad - Z1_tp.grad).abs()
    print(f'Are Z1 grad different?: {torch.any(z1grad_err > tol)}')
    z2grad_err = (Z2.grad - Z2_tp.grad).abs()
    print(f'Are Z2 grad different?: {torch.any(z2grad_err > tol)}')
    a1grad_err = (A1.grad - A1_tp.grad).abs()
    print(f'Are A1 grad different?: {torch.any(a1grad_err > tol)}')

    w1grad_err = (W1.grad - W1_tp.grad).abs()
    print(f'Are W1 grad different?: {torch.any(w1grad_err > tol)}')
    b1grad_err = (b1.grad - b1_tp.grad).abs()
    print(f'Are b1 grad different?: {torch.any(b1grad_err > tol)}')

    w2grad_err = (W2.grad - W2_tp.grad).abs()
    print(f'Are W2 grad different?: {torch.any(w2grad_err > tol)}')
    b2grad_err = (b2.grad - b2_tp.grad).abs()
    print(f'Are b2 grad different?: {torch.any(b2grad_err > tol)}')

#optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) 
#optimizer.step()
#optimizer.zero_grad()
'''