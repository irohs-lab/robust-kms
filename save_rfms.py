import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torchkernels.kernels.radial import laplacian, gaussian
# from radial import laplacian

# Replace laplace_kernel_M with laplacian(X_train, X_test, bandwidth, Mt, False)

import sys
sys.path.append("/users/student/rs/rahulky/robust-rfms/nfa_src/src")
import dataset

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--kernel', type=str, default='laplacian', help='Kernel to use: laplacian or gaussian')
parser.add_argument('--L', type=str)
parser.add_argument('--reg', type=float, default=1e-3, help='Regularization strength for the RFM solver')

#Only to be uncommented while using fmnist
# parser.add_argument('--class1', type=int, default=4, help='Number of class to use from fmnist dataset')
# parser.add_argument('--class2', type=int, default=2, help='Number of class to use from fmnist dataset')
parser.add_argument('--Feature_IDX', type=int, default=31, help='Feature index to use from celeba dataset')

args = parser.parse_args()

# def load_dataset_svhn(NUM_CLASSES=2):

#     return dataset.get_svhn(svhn_class=NUM_CLASSES)

# trainloader, valloader, testloader = load_dataset_svhn(2)

# def ld_fmnist(class1,class2):

#     return dataset.load_fmnist(classes = (class1,class2))

# trainloader, testloader = ld_fmnist(args.class1, args.class2)

def ld_celeba(Feature_IDX):
    return dataset.get_celeba(Feature_IDX)

trainloader, valloader, testloader = ld_celeba(args.Feature_IDX)

# rfm import 
from updated_rfm import *

if args.kernel == "laplacian":
    print("Using Laplacian Kernel")

    Ms, mses , sols, bandwidth, X_train, y_train, best_round_accuracy, iteration, diff_rfm_kernel_acc, kernel_acc  = rfm_laplacian(
        trainloader,
        testloader,
        iters=20,
        loader=True,
        classif=True,
        kernel="laplacian",
        L=float(args.L),
        reg=args.reg,
    )


    # # # Save
    torch.save({
        "Ms": Ms,
        "mses": mses,
        "sols": sols,
        "bandwidth": bandwidth,
        "X_train": X_train,
        "y_train": y_train,
        # "X_test": X_test,
        # "y_test": y_test,
        "best_round_accuracy": best_round_accuracy,
        "iteration": iteration,
        "diff_rfm_kernel_acc": diff_rfm_kernel_acc,
        "kernel_acc": kernel_acc
     }, #f"rfm_results_laplacian_fmnist_L{args.L}_reg{args.reg}_classes({args.class1}_{args.class2}).pt"
     f"rfm_results_laplacian_fmnist_L{args.L}_reg{args.reg}_fidx({args.Feature_IDX}).pt")

elif args.kernel == "gaussian":
    print("Using Gaussian Kernel")
    Ms, mses , sols, bandwidth, X_train, y_train, best_round_accuracy, iteration, diff_rfm_kernel_acc, kernel_acc  = rfm_gaussian(
        trainloader,
        testloader,
        iters=20,
        loader=True,
        classif=True,
        kernel="gaussian",
        L=float(args.L),
        reg=args.reg,
    )

    # # # Save
    torch.save({
        "Ms": Ms,
        "mses": mses,
        "sols": sols,
        "bandwidth": bandwidth,
        "X_train": X_train,
        "y_train": y_train,
        # "X_test": X_test,
        # "y_test": y_test,
        "best_round_accuracy": best_round_accuracy,
        "iteration": iteration,
        "diff_rfm_kernel_acc": diff_rfm_kernel_acc,
        "kernel_acc": kernel_acc
    },  #f"rfm_results_gaussian_fmnist_L{args.L}_reg{args.reg}_classes({args.class1}_{args.class2}).pt"
     f"rfm_results_gaussian_fmnist_L{args.L}_reg{args.reg}_fidx({args.Feature_IDX}).pt")
