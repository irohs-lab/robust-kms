import numpy as np
import torch
from numpy.linalg import solve
import kernels
from tqdm import tqdm
import hickle
import torch.nn.functional as F

from torchkernels.kernels.radial import laplacian, gaussian
# from radial import laplacian

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



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

def normalize(x): return x/x.norm(p=2,dim=-1,keepdim=True)

# def compute_EGOP_with_jacrev_laplacian(X, idx, sol, L, M, num_samples=20000, batch_size=5000):
def compute_EGOP_with_jacrev_laplacian(X, sol, L, M, num_samples=20000, batch_size=5000):
    """
    Compute the full Empirical Gradient Outer Product matrix
    using jacrev for gradient computation
    
    Returns: (d, d) - EGOP matrix
    """
    X = X.to("cpu")
    device, dtype = X.device, X.dtype
    n, d = X.shape
    C = sol.shape[0]
    sol = sol.to(device=device, dtype=dtype)
    M = M.to(device=device, dtype=dtype)
    
    # Sample evaluation points
    if (num_samples is None) or (num_samples >= n):
        x_eval = X
    else:
        idx = torch.randint(0, n, (num_samples,), device=device)
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
    EGOP = torch.zeros(d, d, device='cuda', dtype=torch.float32)
    
    for i in range(0, m, batch_size):
        batch = x_eval[i:i+batch_size]  # (bs, d)
        
        # Compute gradients: (bs, C, d)
        G_batch = torch.nan_to_num(batched_grad_fn(batch), nan = 0.0, posinf = 0.0, neginf = 0.0)

        # return G_batch
        
        # Compute outer products: Σ_c ∇f_c(x) ∇f_c(x)^T for each x
        # G_batch: (bs, C, d) -> (bs, d, C)
        G_T = G_batch.transpose(1, 2)
        
        # (bs, d, C) @ (bs, C, d) -> (bs, d, d)
        outer_products = G_T @ G_batch
        
        # Sum over batch
        EGOP += outer_products.sum(dim=0)
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
    X = X.to("cpu")
    device, dtype = X.device, X.dtype
    n, d = X.shape
    C = sol.shape[0]
    sol = sol.to(device=device, dtype=dtype)
    M = M.to(device=device, dtype=dtype)
    
    # Sample evaluation points
    if (num_samples is None) or (num_samples >= n):
        x_eval = X
    else:
        idx = torch.randint(0, n, (num_samples,), device=device)
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
    EGOP = torch.zeros(d, d, device='cuda', dtype=torch.float32)
    
    for i in range(0, m, batch_size):
        batch = x_eval[i:i+batch_size]  # (bs, d)
        
        # Compute gradients: (bs, C, d)
        G_batch = torch.nan_to_num(batched_grad_fn(batch), nan = 0.0, posinf = 0.0, neginf = 0.0)
        
        # Compute outer products: Σ_c ∇f_c(x) ∇f_c(x)^T for each x
        # G_batch: (bs, C, d) -> (bs, d, C)
        G_T = G_batch.transpose(1, 2)
        
        # (bs, d, C) @ (bs, C, d) -> (bs, d, d)
        outer_products = G_T @ G_batch
        
        # Sum over batch
        EGOP += outer_products.sum(dim=0)
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

    # breakpoint()

    M = np.eye(d, dtype='float32')
    mses = []
    Ms = []
    sols = []
    Ms.append(M+0)

    best_acc = 0

    round_best_acc = -1

    for i in range(iters):
        K_train = laplacian(normalize(X_train), normalize(X_train), L, torch.from_numpy(M)).numpy()
        sol = solve(K_train + reg * np.eye(len(K_train)), y_train).T
        print("Solved:", sol.shape)

        # breakpoint()

        sols.append(sol)
        if train_acc:
            preds = (sol @ K_train).T
            y_pred = torch.from_numpy(preds)
            preds = torch.argmax(y_pred, dim=-1)
            labels = torch.argmax(y_train, dim=-1)
            count = torch.sum(labels == preds).numpy()
            print("Round " + str(i) + " Train Acc: ", (count / len(labels))*100)

        K_test = laplacian(normalize(X_train), normalize(X_test), L, torch.from_numpy(M)).numpy()
        preds = (sol @ K_test).T
#         print("preds",preds)
#         print("y_test",y_test)
        mse_ = ((preds - y_test)**2).mean().item()
        mses.append(mse_)
        print("Round " + str(i) + " MSE: ", mse_)
        
        if classif:
            # y_pred = torch.from_numpy(preds)
            y_pred = preds
            preds = torch.argmax(y_pred, dim=-1)
            labels = torch.argmax(y_test, dim=-1)
            count = torch.sum(labels == preds).numpy()
            print("Round " + str(i) + " Acc: ", count / len(labels))

            if 100*(count/len(labels)) > best_acc:
                best_acc = 100*(count/len(labels))
                round_best_acc = i

            if i == 0:
                kernel_acc = (count / len(labels))*100

        # M1  = get_grads_laplacian(X_train, sol, L, torch.from_numpy(M), batch_size=batch_size).astype('float32')
        M2  = compute_EGOP_with_jacrev_laplacian(X_train, sol, L, torch.from_numpy(M), num_samples=20000).astype('float32')
        # M2 = laplacian_grads_using_jacrev(X_train, sol, L, torch.from_numpy(M), batch_size=batch_size).astype('float32')
        
        # print(torch.allclose(M1, M2, rtol=1e-3, atol=1e-5))
        # breakpoint()

        M = M2

        
        Ms.append(M+0)
        if name is not None:
            hickle.dump(M, 'saved_Ms/M_' + name + '_' + str(i) + '.h')

    # if iters>1:

    K_train = laplacian(normalize(X_train), normalize(X_train), L, torch.from_numpy(M).float()).numpy()
    sol = solve(K_train + reg * np.eye(len(K_train)), y_train).T

    sols.append(sol)
    print("Solved:", sol.shape )

    K_test = laplacian(normalize(X_train), normalize(X_test), L, torch.from_numpy(M).float()).numpy()
    preds = (sol @ K_test).T
    mse = ((preds - y_test)**2).mean().item()
    print("Final MSE: ", mse)
        
    if classif:
        # y_pred = torch.from_numpy(preds)
        y_pred = preds
        preds = torch.argmax(y_pred, dim=-1)
        labels = torch.argmax(y_test, dim=-1)
        count = torch.sum(labels == preds).numpy()
        print(" Final Acc: ", count / len(labels))

        if 100*(count/len(labels)) > best_acc:
            best_acc = 100*(count/len(labels))
            round_best_acc = iters

        diff_acc = best_acc - kernel_acc
        
    return Ms, mses, sols, L, X_train, y_train, best_acc, round_best_acc, diff_acc, kernel_acc

def rfm_gaussian(train_loader, test_loader,
        iters=3, name=None, batch_size=2, reg=1e-3,
        train_acc=False, loader=True, classif=True, kernel='gaussian', L = 2):
    
    if loader:
        print("Loaders provided")
        X_train, y_train = get_data(train_loader)
        X_test, y_test = get_data(test_loader)
    else:
        X_train, y_train = train_loader
        X_test, y_test = test_loader
        
        X_train = torch.from_numpy(X_train).float()
        X_test = torch.from_numpy(X_test).float()
        y_train = torch.from_numpy(y_train).float()
        y_test = torch.from_numpy(y_test).float()
        

    n, d = X_train.shape

    M = np.eye(d, dtype='float32')
    mses = []
    Ms = []
    sols = []
    Ms.append(M+0)

    best_acc = 0

    round_best_acc = -1

    kernel_acc = 0.0

    for i in range(iters):
        K_train = gaussian(normalize(X_train), normalize(X_train), L, torch.from_numpy(M)).numpy()
        sol = solve(K_train + reg * np.eye(len(K_train)), y_train).T
        print("Solved:", sol.shape)

        sols.append(sol)
        if train_acc:
            preds = (sol @ K_train).T
            y_pred = torch.from_numpy(preds)
            preds = torch.argmax(y_pred, dim=-1)
            labels = torch.argmax(y_train, dim=-1)
            count = torch.sum(labels == preds).numpy()
            print("Round " + str(i) + " Train Acc: ", count / len(labels))

        K_test = gaussian(normalize(X_train), normalize(X_test), L, torch.from_numpy(M)).numpy()
        preds = (sol @ K_test).T
#         print("preds",preds)
#         print("y_test",y_test)
        print(type(preds), type(y_test))
        mse_ = np.mean(np.square(preds.numpy() - y_test.numpy()))
        mses.append(mse_)
        print("Round " + str(i) + " MSE: ", mse_)
        print(type(preds), type(y_test))
        
        if classif:
            y_pred = preds
            preds = torch.argmax(y_pred, dim=-1)
            labels = torch.argmax(y_test, dim=-1)
            # labels = y_test
            count = torch.sum(labels == preds).numpy()
            # count = (labels == preds).sum().item()
            print("Round " + str(i) + " Acc: ", (count / len(labels))*100)
            # print("Round " + str(i) + " Acc: ", (count / labels.numel())*100)

            if i == 0:
                kernel_acc = (count / len(labels))*100

            # if 100*(count/labels.numel()) > best_acc:
            if 100*(count/len(labels)) > best_acc:
                best_acc = 100*(count/len(labels))
                round_best_acc = i

        # M1  = get_grads_gaussian(X_train, sol, L, torch.from_numpy(M), batch_size=5000).astype('float32')
        M2 = compute_EGOP_with_jacrev_gaussian(X_train, sol, L, torch.from_numpy(M), num_samples=20000, batch_size=5000).astype('float32')
        
        # print(np.allclose(M1, M2, rtol=1e-3, atol=1e-5))

        # breakpoint()

        M = M2
        
        Ms.append(M+0)
        if name is not None:
            hickle.dump(M, 'saved_Ms/M_' + name + '_' + str(i) + '.h')

    K_train = gaussian(normalize(X_train), normalize(X_train), L, torch.from_numpy(M).float()).numpy()
    sol = solve(K_train + reg * np.eye(len(K_train)), y_train).T

    sols.append(sol)
    print("Solved:", sol.shape )

    K_test = gaussian(normalize(X_train), normalize(X_test), L, torch.from_numpy(M).float()).numpy()
    preds = (sol @ K_test).T
    mse_ = np.mean(np.square(preds.numpy() - y_test.numpy()))
    print("Final MSE: ", mse_)
    
    if classif:
        if iters == 0:
            i = 0
        y_pred = preds
        preds = torch.argmax(y_pred, dim=-1)
        labels = torch.argmax(y_test, dim=-1)
        #labels = y_test
        count = torch.sum(labels == preds).numpy()
        #count = (labels == preds).sum().item()
        print("Round " + str(i) + " Acc: ", (count / len(labels))*100)
        #print("Round " + str(i) + " Acc: ", (count / labels.numel())*100)

        #if 100*(count/labels.numel()) > best_acc:
        if 100*(count/len(labels)) > best_acc:
            best_acc = 100*(count/len(labels))
            round_best_acc = iters
        
        diff_acc = best_acc - kernel_acc
    
        
    return Ms, mses, sols, L, X_train, y_train, best_acc, round_best_acc, diff_acc, kernel_acc
