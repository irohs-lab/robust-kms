import json
import os
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torchkernels.kernels.radial import laplacian, gaussian


import dataset

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--kernel', type=str, default='laplacian', help='Kernel to use: laplacian or gaussian')
parser.add_argument('--L', type=str)
parser.add_argument('--reg', type=float, default=1e-3, help='Regularization strength for the RFM solver')
parser.add_argument(
    '--metrics_only',
    action='store_true',
    help='Only save lightweight metrics JSON (no full tensors).',
)

#Only to be uncommented while using fmnist or svhn class pair
parser.add_argument('--class1', type=int, default =4, help='Number of class to use from fmnist dataset')
parser.add_argument('--class2', type=int, default=2, help='Number of class to use from fmnist dataset')
parser.add_argument('--dataset',type = str)
parser.add_argument('--version', type=str)
parser.add_argument('--mc', action = 'store_true')
#Only to be uncommented while using celeba
# parser.add_argument('--Feature_IDX', type=int, default=38, help='Feature index to use from celeba dataset')

args = parser.parse_args()
# os.makedirs("./svhn2", exist_ok=True)

global trainloader
global valloader
global testloader

if args.dataset == "svhn":

    def load_dataset_svhn(NUM_CLASSES=2):

        return dataset.get_svhn(svhn_class=NUM_CLASSES)

    trainloader, valloader, testloader = load_dataset_svhn(10)


    def load_dataset_svhn(classes):

        return dataset.get_svhn(classes = classes)

    trainloader, valloader, testloader = load_dataset_svhn(classes=(int(args.class1),int(args.class2)))

    print("I am svhn")

elif args.dataset == "fmnist":

    #for two class fmnist
    # def ld_fmnist(class1,class2):
    # For 10-class fmnist
    if args.mc == False:
        def ld_fmnist():

            return dataset.load_fmnist(classes=(int(args.class1),int(args.class2)) , which_mnist='fmnist')
    else:
        def ld_fmnist():

            return dataset.load_fmnist(classes=(int(args.class1),int(args.class2)) , which_mnist='fmnist', multiclass=True)

    trainloader, valloader, testloader = ld_fmnist()

    print("I am FMNIST")

    # print("Loaded FMNIST10 dataset")

elif args.dataset == "qmnist":
    if args.mc == False:
        def ld_fmnist():

            return dataset.load_fmnist(classes=(int(args.class1),int(args.class2)) , which_mnist='qmnist')
    else:
        def ld_fmnist():

            return dataset.load_fmnist(classes=(int(args.class1),int(args.class2)) , which_mnist='qmnist', multiclass=True)

    trainloader, valloader,testloader = ld_fmnist()

    print("I am qmnist")

elif args.dataset == "kmnist":
    if args.mc == False:
        def ld_fmnist():

            return dataset.load_fmnist(classes=(int(args.class1),int(args.class2)) , which_mnist='kmnist')
    else:
        def ld_fmnist():

            return dataset.load_fmnist(classes=(int(args.class1),int(args.class2)) , which_mnist='kmnist', multiclass=True)


    trainloader, valloader,testloader = ld_fmnist()

    print("I am KMNIST")

elif args.dataset == "emnist":
    if args.mc == False:
        def ld_fmnist():

            return dataset.load_fmnist(classes=(int(args.class1),int(args.class2)) , which_mnist='emnist')
    else:
        def ld_fmnist():

            return dataset.load_fmnist(classes=(int(args.class1),int(args.class2)) , which_mnist='emnist', multiclass = True)

    trainloader, valloader, testloader = ld_fmnist()

    print("I am EMNIST")
elif args.dataset == "bank":
    def ld_bank():

        return dataset.uci_bank()

    trainloader, testloader = ld_bank()
    print("I am Bank")

from updated_rfm import *

if args.kernel == "laplacian":
    print("Using Laplacian Kernel")

    Ms, mses , sols, bandwidth, X_train, y_train, best_round_accuracy, iteration, diff_rfm_kernel_acc, kernel_acc  = rfm_laplacian(
        trainloader,
        valloader,
        iters= (5 if args.version == 'rfm' else 0),
        loader=True,
        classif=True,
        kernel="laplacian",
        L=float(args.L),
        reg=args.reg,
    )

    print(f"For reg{args.reg} and L{args.L}\n Best iteration accuracy:", best_round_accuracy, "at iteration:", iteration, "base_kernel_acc:", kernel_acc)


    payload = {
        "best_round_accuracy": best_round_accuracy,
        "iteration": iteration,
        "diff_rfm_kernel_acc": diff_rfm_kernel_acc,
        "kernel_acc": kernel_acc,
        "kernel": args.kernel,
        "L": args.L,
        "reg": args.reg,
        "class1": args.class1,
        "class2": args.class2,
    }
    if args.metrics_only:
        if args.mc == False:
            metrics_path = f"./{args.dataset}2/metrics_{args.kernel}_{args.version}_{args.dataset}_L{args.L}_reg{args.reg}_{args.class1}_{args.class2}.json"
        else:
            metrics_path = f"./{args.dataset}2/metrics_{args.kernel}_{args.version}_{args.dataset}_L{args.L}_reg{args.reg}_mc.json"

        with open(metrics_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(payload, indent=2))
    else:
        payload.update(
            {
                "Ms": Ms,
                "mses": mses,
                "sols": sols,
                "bandwidth": bandwidth,
                "X_train": X_train,
                "y_train": y_train,
            }
        )
        if args.mc == False:
            torch.save(
                payload,
                f"./{args.dataset}2/{args.version}_results_{args.kernel}_{args.dataset}_L{args.L}_reg{args.reg}_{args.class1}_{args.class2}.pt",
                pickle_protocol=4,
            )
        else:
            torch.save(
                payload,
                f"./{args.dataset}2/{args.version}_results_{args.kernel}_{args.dataset}_L{args.L}_reg{args.reg}_mc.pt",
                pickle_protocol=4,
            )


elif args.kernel == "gaussian":
    print("Using Gaussian Kernel")
    Ms, mses , sols, bandwidth, X_train, y_train, best_round_accuracy, iteration, diff_rfm_kernel_acc, kernel_acc  = rfm_gaussian(
        trainloader,
        valloader,
        # For RFMS
        iters = (5 if args.version == "rfm" else 0),
        # iters = 1,
        loader=True,
        classif=True,
        kernel="gaussian",
        L=float(args.L),
        reg=args.reg,
    )

    print(f"For reg{args.reg} and L{args.L}\n Best iteration accuracy:", best_round_accuracy, "at iteration:", iteration, "base_kernel_acc:", kernel_acc)

    payload = {
        "best_round_accuracy": best_round_accuracy,
        "iteration": iteration,
        "diff_rfm_kernel_acc": diff_rfm_kernel_acc,
        "kernel_acc": kernel_acc,
        "kernel": args.kernel,
        "L": args.L,
        "reg": args.reg,
        "class1": args.class1,
        "class2": args.class2,
    }
    if args.metrics_only:
        if args.mc == False:
            metrics_path = f"./{args.dataset}2/metrics_{args.kernel}_{args.version}_{args.dataset}_L{args.L}_reg{args.reg}_{args.class1}_{args.class2}.json"
        else:
            metrics_path = f"./{args.dataset}2/metrics_{args.kernel}_{args.version}_{args.dataset}_L{args.L}_reg{args.reg}_mc.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(payload, indent=2))
    else:
        payload.update(
            {
                "Ms": Ms,
                "mses": mses,
                "sols": sols,
                "bandwidth": bandwidth,
                "X_train": X_train,
                "y_train": y_train,
            }
        )
        if args.mc == False:
            torch.save(
                payload,
                f"./{args.dataset}2/{args.version}_results_{args.kernel}_{args.dataset}_L{args.L}_reg{args.reg}_{args.class1}_{args.class2}.pt",
                pickle_protocol=4,
            )
        else:
            torch.save(
                payload,
                f"./{args.dataset}2/{args.version}_results_{args.kernel}_{args.dataset}_L{args.L}_reg{args.reg}_mc.pt",
                pickle_protocol=4,
            )
