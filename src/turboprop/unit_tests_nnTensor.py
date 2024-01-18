import turboprop as tp
import nnTensor as nnT
import numpy as np
import torch

w1_np = np.random.randn(8, 4 * 4).astype(np.float32)
b1_np = np.random.randn(8, 1).astype(np.float32)
X_np = np.random.randn(4 * 4, 20).astype(np.float32)
y_np = np.random.randint(0, 8, (1, 20))

W1_tp = tp.Tensor(w1_np)
b1_tp = tp.Tensor(b1_np) 
Z_tp = W1_tp @ X_np + b1_tp
#print(Z_tp.array)
A_tp = Z_tp.relu()
loss_tp = nnT.CrossEntropyLoss()(A_tp, y_np)
print(loss_tp)


W1 = torch.tensor(w1_np, requires_grad=True)
b1 = torch.tensor(b1_np, requires_grad=True) 
X = torch.tensor(X_np)  
Z = W1 @ X + b1
Z.retain_grad()
A = torch.nn.ReLU()(Z)
y = torch.tensor(y_np.squeeze())
#loss = torch.nn.CrossEntropyLoss()(Z.T, y)
loss = torch.nn.CrossEntropyLoss()(A.T, y)
print(loss)

## forward pb matching!

## now check grad
loss_tp.backward()
#print(Z_tp.grad.shape)

loss.backward()
#print(Z.grad.shape)


rel_tol = 1e-3
with torch.no_grad():
    z_rel_err = (Z - Z_tp.array).abs() / Z.abs()
    print(f'Are Z fwd different?: {torch.any(z_rel_err > rel_tol)}')
    zgrad_rel_err = (Z.grad - Z_tp.grad).abs() / Z.grad.abs()
    print(f'Are Z grad different?: {torch.any(zgrad_rel_err > rel_tol)}')

    w1grad_rel_err = (W1.grad - W1_tp.grad).abs() / W1.grad.abs()
    print(f'Are W1 grad different?: {torch.any(w1grad_rel_err > rel_tol)}')
    
    b1grad_rel_err = (b1.grad - b1_tp.grad).abs() / b1.grad.abs()
    print(f'Are b1 grad different?: {torch.any(b1grad_rel_err > rel_tol)}')
