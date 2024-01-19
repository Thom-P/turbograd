import turboprop as tp
import nnTensor as nnT
import numpy as np
import torch
torch.set_printoptions(precision=6)

w1_np = np.random.randn(16, 28 * 28).astype(np.float32)
b1_np = np.random.randn(16, 1).astype(np.float32)
w2_np = np.random.randn(10, 16).astype(np.float32)
b2_np = np.random.randn(10, 1).astype(np.float32)
X_np = np.random.randn(28 * 28, 500).astype(np.float32)
y_np = np.random.randint(0, 10, (1, 500))

W1_tp = tp.Tensor(w1_np)
b1_tp = tp.Tensor(b1_np) 
Z1_tp = W1_tp @ X_np + b1_tp
A1_tp = Z1_tp.relu()
W2_tp = tp.Tensor(w2_np)
b2_tp = tp.Tensor(b2_np)
Z2_tp = W2_tp @ A1_tp + b2_tp

loss_tp = nnT.CrossEntropyLoss()(Z2_tp, y_np)
print(loss_tp)

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

'''
rel_tol = 1e-2
with torch.no_grad():
    z1_rel_err = (Z1 - Z1_tp.array).abs() / Z1.abs()
    print(f'Are Z1 fwd different?: {torch.any(z1_rel_err > rel_tol)}')
    a1_rel_err = (A1 - A1_tp.array).abs() / A1.abs()
    print(f'Are A1 fwd different?: {torch.any(a1_rel_err > rel_tol)}')
    z2_rel_err = (Z2 - Z2_tp.array).abs() / Z2.abs()
    print(f'Are Z2 fwd different?: {torch.any(z2_rel_err > rel_tol)}')
    
    z1grad_rel_err = (Z1.grad - Z1_tp.grad).abs() / Z1.grad.abs()
    print(f'Are Z1 grad different?: {torch.any(z1grad_rel_err > rel_tol)}')
    z2grad_rel_err = (Z2.grad - Z2_tp.grad).abs() / Z2.grad.abs()
    print(f'Are Z2 grad different?: {torch.any(z2grad_rel_err > rel_tol)}')
    a1grad_rel_err = (A1.grad - A1_tp.grad).abs() / A1.grad.abs()
    print(f'Are A1 grad different?: {torch.any(a1grad_rel_err > rel_tol)}')

    w1grad_rel_err = (W1.grad - W1_tp.grad).abs() / W1.grad.abs()
    print(f'Are W1 grad different?: {torch.any(w1grad_rel_err > rel_tol)}')
    b1grad_rel_err = (b1.grad - b1_tp.grad).abs() / b1.grad.abs()
    print(f'Are b1 grad different?: {torch.any(b1grad_rel_err > rel_tol)}')

    w2grad_rel_err = (W2.grad - W2_tp.grad).abs() / W2.grad.abs()
    print(f'Are W2 grad different?: {torch.any(w2grad_rel_err > rel_tol)}')
    b2grad_rel_err = (b2.grad - b2_tp.grad).abs() / b2.grad.abs()
    print(f'Are b2 grad different?: {torch.any(b2grad_rel_err > rel_tol)}')

    print(Z2.grad[z2grad_rel_err > rel_tol])
    print(Z2_tp.grad[z2grad_rel_err > rel_tol])
# still some small diffs here and there but probably not important
'''    

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

    #print(A1.grad[a1grad_err > tol])
    #print(A1_tp.grad[a1grad_err > tol])
    #print(A1_tp.grad[a1grad_err > tol].shape)

    #print(A1.grad)
    #print(A1_tp.grad)
    #print(W2_tp.array.T @ Z2_tp.grad)
    #print('***')
    #print(Z1.grad)
    #print(Z1_tp.grad)


   
    #print(W2)
    #print(W2_tp.array)
#print(loss_tp.parameters())

# CURRENT BUG: A1 grad is wrong and is equal to Z1 grad, why? probably mem copy issue in building ops

# bug in steps:

