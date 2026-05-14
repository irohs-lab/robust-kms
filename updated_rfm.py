import numpy as np
import torch
from numpy.linalg import solve
from tqdm import tqdm
import hickle
import torch.nn.functional as F

from torchkernels.kernels.radial import laplacian, gaussian
# from radial import laplacian

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device="cpu"

# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument('--kernel', type=str, default='laplacian', help='Kernel to use: laplacian or gaussian')
# args = parser.parse_args()

# def laplace_kernel_M(pair1, pair2, bandwidth, M):
#     return kernels.laplacian_M(pair1, pair2, bandwidth, M)

from torch.func import vmap, jacrev
from functools import partial

def get_data(loader):
    X = []
    y = []
    for idx, batch in enumerate(loader):
        inputs, labels = batch
        X.append(inputs)
        y.append(labels)
    return torch.cat(X, dim=0), torch.cat(y, dim=0)

def normalize(x): return x/x.norm(p=2,dim=-1,keepdim=True).clamp_min(1e-12)

def get_fixed_indices(n, num_samples, seed=42, device="cpu"):
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    return torch.randint(0, n, (num_samples,), generator=g, device=device)

# def compute_EGOP_with_jacrev_laplacian(X, idx, sol, L, M, num_samples=20000, batch_size=5000):
def compute_EGOP_with_jacrev_laplacian(X, sol, L, M, num_samples=20000, batch_size=5000):
    """
    Compute the full Empirical Gradient Outer Product matrix
    using jacrev for gradient computation
    
    Returns: (d, d) - EGOP matrix
    """
    # batch_size = 5000
    X = X.to(dev)
    device, dtype = X.device, X.dtype
    n, d = X.shape
    C = sol.shape[0]
    sol = sol.to(device=device, dtype=dtype)
    M = M.to(device=device, dtype=dtype)
    
    # Sample evaluation points
    if (num_samples is None) or (num_samples >= n):
        x_eval = X
    else:
        idx = get_fixed_indices(n, num_samples, seed=42, device=device)
        x_eval = X[idx]
    m = x_eval.shape[0]
    
    # Define the vectorized gradient function
    def f_all_classes(x):
        x_hat = normalize(x.unsqueeze(0))
        X_hat = normalize(X)
        K = laplacian(X_hat, x_hat, L, M, in_place=False)  # (n, 1)
        return (sol @ K).squeeze()
    
    grad_fn = jacrev(f_all_classes)
    batched_grad_fn = vmap(grad_fn)
    
    # Compute EGOP in batches
    # EGOP = torch.zeros(d, d, device='cuda', dtype=torch.float32)
    EGOP = torch.zeros(d, d, device=dev, dtype=torch.float32)

    
    for i in range(0, m, batch_size):
        batch = x_eval[i:i+batch_size]  # (bs, d)
        
        # Compute gradients: (bs, C, d)
        G_batch = torch.nan_to_num(batched_grad_fn(batch), nan = 0.0, posinf = 0.0, neginf = 0.0)

        # return G_batch
        
        # Compute outer products: Σ_c ∇f_c(x) ∇f_c(x)^T for each x
        # G_batch: (bs, C, d) -> (bs, d, C)
        # G_T = G_batch.transpose(1, 2)
        
        # # (bs, d, C) @ (bs, C, d) -> (bs, d, d)
        # outer_products = G_T @ G_batch
        
        # # Sum over batch
        # EGOP += outer_products.sum(dim=0)

        EGOP += torch.einsum('bcd,bce->de', G_batch, G_batch)

        print(f"Processed batch {i//batch_size + 1}/{(m + batch_size - 1)//batch_size}")
    
    EGOP /= m
    EGOP = EGOP.to('cpu')
    return EGOP.numpy()

def compute_EGOP_with_jacrev_gaussian(X, sol, L, M, num_samples=20000, batch_size=2):
    """
    Compute the full Empirical Gradient Outer Product matrix
    using jacrev for gradient computation
    
    Returns: (d, d) - EGOP matrix
    """
    # batch_size = 5000
    X = X.to(dev)
    device, dtype = X.device, X.dtype
    n, d = X.shape
    C = sol.shape[0]
    sol = sol.to(device=device, dtype=dtype)
    M = M.to(device=device, dtype=dtype)
    
    # Sample evaluation points
    if (num_samples is None) or (num_samples >= n):
        x_eval = X
    else:
        idx = get_fixed_indices(n, num_samples, seed=42, device=device)
        x_eval = X[idx]
    m = x_eval.shape[0]
    
    # Define the vectorized gradient function
    def f_all_classes(x):
        x_hat = normalize(x.unsqueeze(0))
        X_hat = normalize(X)
        K = gaussian(X_hat, x_hat, L, M, in_place=False)  # (n, 1)
        return (sol @ K).squeeze()
    
    grad_fn = jacrev(f_all_classes)
    batched_grad_fn = vmap(grad_fn)
    
    # Compute EGOP in batches
    # EGOP = torch.zeros(d, d, device='cuda', dtype=torch.float32)
    EGOP = torch.zeros(d, d, device=dev, dtype=torch.float32)
    
    for i in range(0, m, batch_size):
        batch = x_eval[i:i+batch_size]  # (bs, d)
        
        # Compute gradients: (bs, C, d)
        G_batch = torch.nan_to_num(batched_grad_fn(batch), nan = 0.0, posinf = 0.0, neginf = 0.0)
        
        EGOP += torch.einsum('bcd,bce->de', G_batch, G_batch)

        print(f"Processed batch {i//batch_size + 1}/{(m + batch_size - 1)//batch_size}")
    
    EGOP /= m
    EGOP = EGOP.to('cpu')
    return EGOP.numpy()

def rfm_laplacian(train_loader, test_loader,
        iters=3, name=None, batch_size=5000, reg=1e-3,
        train_acc=False, loader=True, classif=True, kernel='laplacian', L = 10):
    
    if loader:
        print("Loaders provided")
        X_train, y_train = get_data(train_loader)
        X_test, y_test = get_data(test_loader)
        # y_mean = y_train.mean(dim=0, keepdim=True)
        # print("y_mean:", y_mean)
        # y_train = y_train - y_mean
    else:
        X_train, y_train = train_loader
        X_test, y_test = test_loader
        
        X_train = torch.from_numpy(X_train).float()
        X_test = torch.from_numpy(X_test).float()
        y_train = torch.from_numpy(y_train).float()
        y_test = torch.from_numpy(y_test).float()
        
    n, d = X_train.shape

    print("n:", n, "d:", d)
    print("X_train[0]:", X_train[0].shape)
    print("y_train:", y_train.shape)

    # Move once to GPU
    X_train = X_train.to(dev)
    X_test = X_test.to(dev)
    y_train = y_train.to(dev)
    y_test = y_test.to(dev)

    M = torch.eye(d, device=dev, dtype=torch.float32)
    mses = []
    Ms = []
    sols = []
    Ms.append(M.detach().cpu().numpy().copy())

    best_acc = 0.0
    round_best_acc = -1
    kernel_acc = 0.0

    # Precompute normalized test once
    Xh_test = normalize(X_test)

    for i in range(iters):
        # GPU kernel matrix + GPU solve
        Xh_train = normalize(X_train)
        K_train = laplacian(Xh_train, Xh_train, L, M)
        I = torch.eye(K_train.shape[0], device=dev, dtype=K_train.dtype)
        sol = torch.linalg.solve(K_train + reg * I, y_train).T
        print("Solved:", sol.shape)

        sols.append(sol.detach().cpu().numpy())

        if train_acc:
            preds = (sol @ K_train).T
            preds_idx = torch.argmax(preds, dim=-1)
            labels = y_train.argmax(dim=-1) if y_train.ndim > 1 else y_train.long()
            count = torch.sum(labels == preds_idx).item()
            print("Round " + str(i) + " Train Acc: ", (count / len(labels)) * 100)

        # GPU test kernel + prediction
        K_test = laplacian(Xh_train, Xh_test, L, M)
        preds = (sol @ K_test).T

        mse_ = ((preds - y_test) ** 2).mean().item()
        mses.append(mse_)
        print("Round " + str(i) + " MSE: ", mse_)
        
        if classif:
            y_pred = preds
            preds_idx = torch.argmax(y_pred, dim=-1)
            labels = y_test.argmax(dim=-1) if y_test.ndim > 1 else y_test.long()
            count = torch.sum(labels == preds_idx).item()
            print("Round " + str(i) + " Acc: ", count / len(labels))

            if 100 * (count / len(labels)) > best_acc:
                best_acc = 100 * (count / len(labels))
                round_best_acc = i

            if i == 0:
                kernel_acc = (count / len(labels)) * 100

        # GPU EGOP update
        M2 = compute_EGOP_with_jacrev_laplacian(
            X_train, sol, L, M, num_samples=20000, batch_size=2500
        ).astype('float32')

        M = torch.from_numpy(M2).to(dev)

        Ms.append(M.detach().cpu().numpy().copy())
        if name is not None:
            hickle.dump(M.detach().cpu().numpy(), 'saved_Ms/M_' + name + '_' + str(i) + '.h')

    # Final GPU solve
    Xh_train = normalize(X_train)
    K_train = laplacian(Xh_train, Xh_train, L, M)
    I = torch.eye(K_train.shape[0], device=dev, dtype=K_train.dtype)
    sol = torch.linalg.solve(K_train + reg * I, y_train).T

    sols.append(sol.detach().cpu().numpy())
    print("Solved:", sol.shape)

    K_test = laplacian(Xh_train, Xh_test, L, M)
    preds = (sol @ K_test).T
    mse = ((preds - y_test) ** 2).mean().item()
    print("Final MSE: ", mse)
        
    if classif:
        y_pred = preds
        preds_idx = torch.argmax(y_pred, dim=-1)
        labels = y_test.argmax(dim=-1) if y_test.ndim > 1 else y_test.long()
        count = torch.sum(labels == preds_idx).item()
        print(" Final Acc: ", count / len(labels))

        if 100 * (count / len(labels)) > best_acc:
            best_acc = 100 * (count / len(labels))
            round_best_acc = iters

        diff_acc = best_acc - kernel_acc
        
    return Ms, mses, sols, L, X_train.detach().cpu(), y_train.detach().cpu(), best_acc, round_best_acc, diff_acc, kernel_acc

def rfm_gaussian(train_loader, test_loader,
        iters=3, name=None, batch_size=2, reg=1e-3,
        train_acc=False, loader=True, classif=True, kernel='gaussian', L = 2):
    
    if loader:
        print("Loaders provided")
        X_train, y_train = get_data(train_loader)
        X_test, y_test = get_data(test_loader)
        # y_mean = y_train.mean(dim=0, keepdim=True)
        # print("y_mean:", y_mean)
        # y_train = y_train - y_mean   # (1, C)
    else:
        X_train, y_train = train_loader
        X_test, y_test = test_loader
        
        X_train = torch.from_numpy(X_train).float()
        X_test = torch.from_numpy(X_test).float()
        y_train = torch.from_numpy(y_train).float()
        y_test = torch.from_numpy(y_test).float()
        

    n, d = X_train.shape

    # Move once to GPU
    X_train = X_train.to(dev)
    X_test = X_test.to(dev)
    y_train = y_train.to(dev)
    y_test = y_test.to(dev)

    M = torch.eye(d, device=dev, dtype=torch.float32)
    mses = []
    Ms = []
    sols = []
    Ms.append(M.detach().cpu().numpy().copy())

    best_acc = 0.0

    round_best_acc = -1

    kernel_acc = 0.0

    # Precompute normalized test once
    Xh_test = normalize(X_test)

    for i in range(iters):
        Xh_train = normalize(X_train)
        K_train = gaussian(Xh_train, Xh_train, L, M)
        I = torch.eye(K_train.shape[0], device=dev, dtype=K_train.dtype)
        sol = torch.linalg.solve(K_train + reg * I, y_train).T
        print("Solved:", sol.shape)

        sols.append(sol.detach().cpu().numpy())

        if train_acc:
            preds = (sol @ K_train).T
            preds_idx = torch.argmax(preds, dim=-1)
            labels = y_train.argmax(dim=-1) if y_train.ndim > 1 else y_train.long()
            count = torch.sum(labels == preds_idx).item()
            print("Round " + str(i) + " Train Acc: ", count / len(labels))

        K_test = gaussian(Xh_train, Xh_test, L, M)
        preds = (sol @ K_test).T

        mse_ = ((preds - y_test) ** 2).mean().item()
        mses.append(mse_)
        print("Round " + str(i) + " MSE: ", mse_)
        
        if classif:
            y_pred = preds
            preds_idx = torch.argmax(y_pred, dim=-1)
            labels = y_test.argmax(dim=-1) if y_test.ndim > 1 else y_test.long()
            count = torch.sum(labels == preds_idx).item()
            print("Round " + str(i) + " Acc: ", (count / len(labels)) * 100)

            if i == 0:
                kernel_acc = (count / len(labels)) * 100

            if 100 * (count / len(labels)) > best_acc:
                best_acc = 100 * (count / len(labels))
                round_best_acc = i

        M2 = compute_EGOP_with_jacrev_gaussian(
            X_train, sol, L, M, num_samples=20000, batch_size=2500
        ).astype('float32')

        M = torch.from_numpy(M2).to(dev)

        # Checking normalization.
        # M / (M.max() + 1e-30)


        Ms.append(M.detach().cpu().numpy().copy())
        if name is not None:
            hickle.dump(M.detach().cpu().numpy(), 'saved_Ms/M_' + name + '_' + str(i) + '.h')

    Xh_train = normalize(X_train)
    K_train = gaussian(Xh_train, Xh_train, L, M)
    I = torch.eye(K_train.shape[0], device=dev, dtype=K_train.dtype)
    sol = torch.linalg.solve(K_train + reg * I, y_train).T

    sols.append(sol.detach().cpu().numpy())
    print("Solved:", sol.shape)

    K_test = gaussian(Xh_train, Xh_test, L, M)
    preds = (sol @ K_test).T
    mse_ = ((preds - y_test) ** 2).mean().item()
    print("Final MSE: ", mse_)
    
    if classif:
        if iters == 0:
            i = 0
        y_pred = preds
        preds_idx = torch.argmax(y_pred, dim=-1)
        labels = y_test.argmax(dim=-1) if y_test.ndim > 1 else y_test.long()
        count = torch.sum(labels == preds_idx).item()
        print("Final Acc: ", (count / len(labels)) * 100)

        if 100 * (count / len(labels)) > best_acc:
            best_acc = 100 * (count / len(labels))
            round_best_acc = iters
        
        diff_acc = best_acc - kernel_acc
    
        
    return Ms, mses, sols, L, X_train.detach().cpu(), y_train.detach().cpu(), best_acc, round_best_acc, diff_acc, kernel_acc
