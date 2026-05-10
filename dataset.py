import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
from numpy.linalg import norm
import os
from torchvision.datasets import CelebA
import pandas as pd
import pickle

import matplotlib.pyplot as plt

import random

def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class CelebA_NoHash(CelebA):
    def _check_integrity(self):
        # only require that the files exist
        files = [
            'img_align_celeba',          # folder
            'list_attr_celeba.txt',
            'list_eval_partition.txt'
        ]
        root = os.path.join(self.root, self.base_folder)
        return all(os.path.exists(os.path.join(root, f)) for f in files)

def convert_to_greyscale_and_flatten(dataloader):
    conv_x = []
    conv_y = []
    weights = torch.tensor([0.2989, 0.5870, 0.1140]).view(3, 1, 1)

    for x, y in dataloader:
        for k in x:
            k = k.reshape(3, 32, 32)
            gray = (k * weights).sum(dim=0, keepdim=True)
            conv_x.append(gray.flatten())

        for k in y:
            # If even then [1,0] else [0,1]
            if k[0] != 0 or k[2] != 0 or k[4] != 0 or k[6] != 0 or k[8] != 0:
                conv_y.append([1, 0])
            else:
                conv_y.append([0, 1])

    conv_x = torch.stack(conv_x)
    conv_y = torch.tensor(conv_y)

    return DataLoader(TensorDataset(conv_x, conv_y), batch_size=128, shuffle=True)

def one_hot_data(dataset, num_classes, num_samples=-1, shift_label=False):
    
    for images, labels in dataset:      
        print("Images shape:", images.shape)

    #Dictionary to hold one-hot encoded labels
    labelset = {}

    for i in range(num_classes):

        # Create a one-hot encoded vector for each class
        one_hot = torch.zeros(num_classes)

        # Set the index corresponding to the class to 1
        one_hot[i] = 1

        # Add the one-hot vector to the labelset dictionary
        labelset[i] = one_hot

    offset = 0
    if shift_label:
        offset = -1

    subset = [(ex.flatten(), labelset[label + offset]) \
              for idx, (ex, label) in enumerate(dataset) if idx < num_samples]

    return subset


# def split(trainset, p=.8):
#     train, val = train_test_split(trainset, train_size=p)
#     return train, val

def split(trainset, p=0.8, seed=42, stratify_labels=None):
    train, val = train_test_split(
        trainset,
        train_size=p,
        random_state=seed,
        stratify=stratify_labels
    )
    return train, val

def get_dataset_path():
    """
    Returns the absolute path to the dataset directory.
    The path will be: <home_directory>/Ansatz/deep_neual_feature_ansatz/datasets
    If the directory does not exist, it will be created.
    """
    # Construct the relative path from the home directory
    # relative_path = os.path.join('robust-rfms', 'nfa_src', 'datasets')
    
    # Expand the ~ to the actual home directory path
    # dataset_dir = os.path.expanduser(os.path.join('~', relative_path))
    dataset_dir = os.path.join('/janaki', 'common', 'Datasets', 'rahulky')
    
    # Create the directory if it does not exist
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
    
    return dataset_dir

# For pytorch1.13.2
# def get_svhn(split_percentage=.8, num_train=np.float('inf'), num_test=np.float('inf'), svhn_class=10):

def to_dense(x):
    return x.toarray() if hasattr(x, "toarray") else np.asarray(x)

def save_svhn_grid(loader, num_images=10, save_path="outputs/svhn_grid.png", classes=None):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    X, y = next(iter(loader))

    # Works for both one-hot binary and plain multiclass labels
    if y.ndim == 2:
        labels = torch.argmax(y, dim=1)
        if classes is not None:
            labels = torch.tensor([classes[i.item()] for i in labels])
    else:
        labels = y.long()

    fig, axes = plt.subplots(1, num_images, figsize=(15, 3))

    for i in range(num_images):
        img = X[i]

        # binary loader gives flattened image: (3072,)
        if img.ndim == 1:
            img = img.view(3, 32, 32)

        img = img.permute(1, 2, 0).cpu().clamp(0, 1)

        axes[i].imshow(img)
        axes[i].set_title(str(labels[i].item()))
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved to: {save_path}")

# For ker_rfm
# def get_svhn(split_percentage=.8, num_train=np.float64('inf'), num_test=np.float64('inf'), svhn_class=10, classes = (0,1), batch_size=128):
def get_svhn(split_percentage=.8, svhn_class=10, classes = (0,1), batch_size=128):
    NUM_CLASSES = 10
    transform = transforms.Compose([transforms.ToTensor()])
#    svhn_path = '~/datasets/'
    svhn_path = get_dataset_path()

    trainset = torchvision.datasets.SVHN(root=svhn_path,
                                         split='train',
                                         transform=transform,
                                         download=True)
    
    print("Trainset:", trainset)

    testset = torchvision.datasets.SVHN(root=svhn_path,
                                        split='test',
                                        transform=transform,
                                        download=True)
    
    # trainset = one_hot_data(trainset, NUM_CLASSES, num_samples=num_train)
    trainset, valset = split(trainset, p=split_percentage)

    # trainset/valset are lists of (x, y)
    X_train = torch.stack([x for x, _ in trainset]).flatten(1)
    y_train = torch.tensor([y for _, y in trainset])

    X_val = torch.stack([x for x, _ in valset]).flatten(1)
    y_val = torch.tensor([y for _, y in valset])

    X_test = torch.stack([x for x, _ in testset]).flatten(1)
    y_test = torch.tensor([y for _, y in testset])

    print("TestSet:", len(testset))
    print((X_test).max())
    print(max(y_test))

    # breakpoint()

    if classes is None:

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                                shuffle=True, num_workers=1)
        
        if svhn_class!= 10:
            trainloader = convert_to_greyscale_and_flatten(trainloader)

        valloader = torch.utils.data.DataLoader(valset, batch_size=128,
                                                shuffle=False, num_workers=1)
        
        if svhn_class != 10:
            valloader = convert_to_greyscale_and_flatten(valloader)

        # testset = one_hot_data(testset, NUM_CLASSES, num_samples=num_test)
        testset = one_hot_data(testset, NUM_CLASSES, num_samples=float('inf'))


        testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                                shuffle=False, num_workers=1)
        
        if svhn_class != 10:
            testloader = convert_to_greyscale_and_flatten(testloader)

        print("Num Train: ", len(trainset), "Num Val: ", len(valset),
            "Num Test: ", len(testset))
        
    else:
        classes = tuple(classes)
        if len(classes) != 2:
            raise ValueError("`classes` must contain exactly two class labels.")
        if classes[0] == classes[1]:
            raise ValueError("`classes` must refer to two distinct classes.")

        counts = torch.bincount(y_train, minlength=10)
        for cls, n in enumerate(counts.tolist()):
            print(cls, n)

        c0, c1 = classes
        train_mask = (y_train == c0) | (y_train == c1)
        test_mask = (y_test == c0) | (y_test == c1)
        val_mask = (y_val == c0) | (y_val == c1)

        X_train = X_train[train_mask]
        y_train = y_train[train_mask]
        
        X_test = X_test[test_mask]
        y_test = y_test[test_mask]

        X_val = X_val[val_mask]
        y_val = y_val[val_mask]

        y_train = torch.where(y_train == c0, torch.zeros_like(y_train), torch.ones_like(y_train))
        y_val = torch.where(y_val == c0, torch.zeros_like(y_val), torch.ones_like(y_val))
        y_test = torch.where(y_test == c0, torch.zeros_like(y_test), torch.ones_like(y_test))

        print("Filtered train dataset size:", X_train.shape)
        print("Filtered test dataset size:", X_test.shape)
        print("Filtered Validation dataset size:", X_val.shape)


        print("Filtered y_train shape:", y_train.shape)
        print("Filtered y_test shape:", y_test.shape)
        print("Filtered Validation set shape:", y_val.shape)

        y_train = F.one_hot(y_train.to(torch.long), num_classes=2).float()
        y_test = F.one_hot(y_test.to(torch.long), num_classes=2).float()
        y_val = F.one_hot(y_val.to(torch.long), num_classes=2).float()

        train_tensor_dataset = TensorDataset(X_train, y_train)
        Val_tensor_dataset = TensorDataset(X_val, y_val)
        test_tensor_dataset = TensorDataset(X_test, y_test)

        print("Filtered train dataset size:", X_train.shape)
        print("Filtered test dataset size:", X_test.shape)
        print("Filtered Validation dataset size:", X_val.shape)


        print("Filtered y_train shape:", y_train.shape)
        print("Filtered y_test shape:", y_test.shape)
        print("Filtered Validation set shape:", y_val.shape)
        print((torch.argmax(y_train, dim = 1) == 1).sum())
        print((torch.argmax(y_train, dim = 1) == 0).sum())

        trainloader = DataLoader(train_tensor_dataset, batch_size=batch_size, shuffle=True)
        valloader = DataLoader(Val_tensor_dataset, batch_size=batch_size, shuffle=False )
        testloader = DataLoader(test_tensor_dataset, batch_size=batch_size, shuffle=False)

    save_svhn_grid(trainloader, num_images=10, save_path="outputs/svhn_grid.png")
    return trainloader, valloader, testloader 

# get_svhn(classes = None)

# For pytorch1.13.2
# def get_cifar(split_percentage=.8, num_train=np.float('inf'), num_test=np.float('inf')):

def random_subset_indices(total_size, subset_size, seed=42):
    if subset_size > total_size:
        raise ValueError("subset_size cannot be larger than total dataset size")

    rng = np.random.default_rng(seed)
    return rng.choice(total_size, size=subset_size, replace=False)

def print_class_counts(y, split_name="dataset"):
    if y.ndim > 1:
        y_labels = y.argmax(dim=1)
    else:
        y_labels = y

    unique, counts = torch.unique(y_labels, return_counts=True)
    print(f"Class counts in {split_name}:")
    for cls, cnt in zip(unique.tolist(), counts.tolist()):
        print(f"  class {cls}: {cnt}")

def load_fmnist(device='cpu', classes=(0, 1), batch_size=128, which_mnist = 'qmnist', multiclass = False):
    """
    Load Fashion-MNIST and return DataLoaders restricted to a pair of classes.
    """
    transform = transforms.Compose([transforms.ToTensor()])
    root_dir = get_dataset_path()

    # g = torch.Generator()
    # g.manual_seed(42)

    if which_mnist == "fmnist":
        print("Hii, i am FMNIST")
        train_dataset = torchvision.datasets.FashionMNIST(
            root=root_dir, train=True, transform=transform, download=True
        )
        test_dataset = torchvision.datasets.FashionMNIST(
            root=root_dir, train=False, transform=transform, download=True
        )

    elif which_mnist == "kmnist":
        print("Hii, I am KMNIST")
        train_dataset = torchvision.datasets.KMNIST(root= root_dir, train=True, download=True, transform=transform)
        test_dataset  = torchvision.datasets.KMNIST(root= root_dir, train=False, download=True, transform=transform)
    elif which_mnist == "qmnist":
        print("Hii, I am QMNIST")
        train_dataset = torchvision.datasets.QMNIST(root= root_dir, what = "train", compat = True, download=True, transform=transform)
        test_dataset  = torchvision.datasets.QMNIST(root= root_dir, what = "test", compat = True, download=True, transform=transform)
    elif which_mnist == "emnist":
        print("Hii, I am EMNIST")

        train_dataset = torchvision.datasets.EMNIST(
            root=root_dir,
            split="digits",
            train=True,
            download=True
        )

        test_dataset = torchvision.datasets.EMNIST(
            root=root_dir,
            split="digits",
            train=False,
            download=True
        )

    print("Original train dataset size:", len(train_dataset))

        # ---- NEW: randomly select 25,000 samples ----
#     if multiclass == True:
#         subset_size = 25000
#         all_indices = np.arange(len(train_dataset))
#         subset_indices = random_subset_indices(len(train_dataset), subset_size, seed=42)

#         # ---- split ONLY the subset ----
#         # train_idx, val_idx = train_test_split(
#         #     subset_indices, train_size=0.8, random_state=42,  stratify=y_all[subset_indices]
#         train_idx, val_idx = train_test_split(
#             subset_indices, train_size=0.8, random_state=42,  stratify=None
# )

    # For full dataset split
 
    indices = np.arange(len(train_dataset))
    # train_dataset, val_dataset = split(train_dataset, p=0.8)
    train_idx, val_idx = train_test_split(indices, train_size=0.8, random_state=42, stratify=None)

    X_all = train_dataset.data
    y_all = train_dataset.targets

    # X_train = train_dataset.data.to(device).flatten(1).float() / 255.0
    # X_test = test_dataset.data.to(device).flatten(1).float() / 255.0
    # y_train = train_dataset.targets.to(device)
    # y_test = test_dataset.targets.to(device)

    X_train = X_all[train_idx]
    y_train = y_all[train_idx]
    X_val = X_all[val_idx]
    y_val = y_all[val_idx]
    X_test  = test_dataset.data
    y_test  = test_dataset.targets

    # X_train = train_dataset.data
    # X_val = val_dataset.data
    # y_train = train_dataset.targets
    # y_val = val_dataset.targets

    # QMNIST targets are Nx2, label is first column
    if which_mnist == "qmnist":
        y_train = y_train[:, 0]
        y_val = y_val[:, 0]
        y_test  = y_test[:, 0]

    # ---- Fix EMNIST orientation manually (Option A) ----
    # Equivalent to: rot90 + horizontal flip in your transform pipeline
    if which_mnist == "emnist":
        # dataset.data is [N, 28, 28] (uint8)
        X_train = torch.rot90(X_train, 1, dims=(1, 2))
        X_train = torch.flip(X_train, dims=(2,))
        X_val = torch.rot90(X_val, 1, dims=(1, 2))
        X_val = torch.flip(X_val, dims=(2,))
        X_test = torch.rot90(X_test, 1, dims=(1, 2))
        X_test = torch.flip(X_test, dims=(2,))

    # ---- Move to device and scale to [0, 1] ----
    X_train = X_train.to(device).flatten(1).float().div_(255.0)
    X_val = X_val.to(device).flatten(1).float().div_(255.0)
    X_test  = X_test.to(device).flatten(1).float().div_(255.0)
    y_train = y_train.to(device)
    y_val = y_val.to(device)
    y_test  = y_test.to(device)

    # print(type(X_train), type(y_train))

    # print(X_train.shape, y_train.shape)
    # print(X_test.shape, y_test.shape)

    # print(X_train[0])
    # print(y_train[0])

    # print(type(X_train), type(y_train))

    # print(X_train.shape, y_train.shape)
    # print(X_test.shape, y_test.shape)

    # breakpoint()

    # Uncomment the commented lines for 2-class filtering 
    if multiclass == False:
        if classes is not None:
            classes = tuple(classes)
            if len(classes) != 2:
                raise ValueError("`classes` must contain exactly two class labels.")
            if classes[0] == classes[1]:
                raise ValueError("`classes` must refer to two distinct classes.")

            c0, c1 = classes
            train_mask = (y_train == c0) | (y_train == c1)
            val_mask = (y_val == c0) | (y_val == c1)
            test_mask = (y_test == c0) | (y_test == c1)

            X_train = X_train[train_mask]
            y_train = y_train[train_mask]
            X_val = X_val[val_mask]
            y_val = y_val[val_mask]
            X_test = X_test[test_mask]
            y_test = y_test[test_mask]

            y_train = torch.where(y_train == c0, torch.zeros_like(y_train), torch.ones_like(y_train))
            y_val = torch.where(y_val == c0, torch.zeros_like(y_val), torch.ones_like(y_val))
            y_test = torch.where(y_test == c0, torch.zeros_like(y_test), torch.ones_like(y_test))

            y_train = F.one_hot(y_train.to(torch.long), num_classes=2).float()
            y_val = F.one_hot(y_val.to(torch.long), num_classes=2).float()
            y_test = F.one_hot(y_test.to(torch.long), num_classes=2).float()

    # For 10 class
    if multiclass == True:
        y_train = F.one_hot(y_train.to(torch.long), num_classes=10).float()
        y_val = F.one_hot(y_val.to(torch.long), num_classes=10).float()
        y_test = F.one_hot(y_test.to(torch.long), num_classes=10).float()



    # print("Filtered train dataset size:", X_train.shape)
    # print("Filtered Validation dataset size:", X_val.shape)
    # print("Filtered test dataset size:", X_test.shape)
    # print("Filtered y_train shape:", y_train.shape)
    # print("Filtered y_val shape:", y_val.shape)
    # print("Filtered y_test shape:", y_test.shape)

    print_class_counts(y_train, "train")
    print_class_counts(y_val, "validation")
    print_class_counts(y_test, "test")

    # print("Here is the sum:", sum(y_train.argmax(dim=1) == 1))
    # print("Filtered y_train shape:", y_train.shape)
    # print("Filtered y_test shape:", y_test.shape)

    train_tensor_dataset = TensorDataset(X_train, y_train)
    val_tensor_dataset = TensorDataset(X_val, y_val)
    test_tensor_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_tensor_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_tensor_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_tensor_dataset, batch_size=batch_size, shuffle=False)

    # breakpoint()

    return train_loader, val_loader, test_loader

def uci_bank(batch_size=128):
    with open("data.pkl", "rb") as f:
        X_train_proc, X_test_proc, y_train, y_test = pickle.load(f)

    X_train_np = to_dense(X_train_proc).astype(np.float32)
    X_test_np  = to_dense(X_test_proc).astype(np.float32)

    # ---- Convert to one-hot here ----
    y_train_np = np.asarray(y_train).astype(np.int64)
    y_test_np  = np.asarray(y_test).astype(np.int64)

    num_classes = 2
    y_train_np = np.eye(num_classes)[y_train_np].astype(np.float32)
    y_test_np  = np.eye(num_classes)[y_test_np].astype(np.float32)
    # ---------------------------------

    train_ds = TensorDataset(
        torch.from_numpy(X_train_np),
        torch.from_numpy(y_train_np)
    )

    test_ds = TensorDataset(
        torch.from_numpy(X_test_np),
        torch.from_numpy(y_test_np)
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, drop_last=False)

    return train_loader, test_loader
# For ker_rfm
def get_cifar(split_percentage=.8, num_train=np.float64('inf'), num_test=np.float64('inf')):

    NUM_CLASSES = 10
    transform = transforms.Compose([transforms.ToTensor()])
#    path = '~/datasets/'
    cifar_path = get_dataset_path()

    trainset = torchvision.datasets.CIFAR10(root=cifar_path,
                                            train=True,
                                            transform=transform,
                                            download=True)

    trainset = one_hot_data(trainset, NUM_CLASSES, num_samples=num_train)
    trainset, valset = split(trainset, p=split_percentage)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                              shuffle=True, num_workers=1)

    valloader = torch.utils.data.DataLoader(valset, batch_size=64,
                                            shuffle=False, num_workers=1)

    testset = torchvision.datasets.CIFAR10(root=cifar_path,
                                           train=False,
                                           transform=transform,
                                           download=True)

    testset = one_hot_data(testset, NUM_CLASSES, num_samples=num_test)

    testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                             shuffle=False, num_workers=1)

    print("Num Train: ", len(trainset), "Num Val: ", len(valset),
          "Num Test: ", len(testset))
    return trainloader, valloader, testloader


def sample_data(num, d):
    X = np.random.normal(size=(num, d))
    y = X[:, 0] * X[:, 1]
    y = y.reshape(-1, 1)
    return torch.from_numpy(X).float(), torch.from_numpy(y).float()


def get_two_coordinates(split_percentage=.8, num_train=2000, num_test=1000, d=100):
    X, y = sample_data(num_train, d)
    trainset = list(zip(X, y))
    trainset, valset = split(trainset, p=split_percentage)
    X_test, y_test = sample_data(num_test, d)
    testset = list(zip(X_test, y_test))

    train_loader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=1)
    val_loader = DataLoader(valset, batch_size=128, shuffle=False, num_workers=1)
    test_loader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=1)
    print("Num Train: ", len(trainset), "Num Val: ", len(valset),
          "Num Test: ", len(testset))

    return train_loader, val_loader, test_loader


def celeba_subset(dataset, feature_idx, index_list = None, num_samples=-1):

    NUM_CLASSES = 2
    labelset = {}
    for i in range(NUM_CLASSES):
        one_hot = torch.zeros(NUM_CLASSES)
        one_hot[i] = 1
        labelset[i] = one_hot

    by_class = {}
    features = []
    count = 0  # counter for processed items
    for idx in tqdm(index_list):
        ex, label = dataset[idx]
        features.append(label[feature_idx])
        g = label[feature_idx].numpy().item()
        #ex = torch.mean(ex, dim=0)
        ex = ex.flatten()
        ex = ex / norm(ex)
        if g in by_class:
            by_class[g].append((ex, labelset[g]))
        else:
            by_class[g] = [(ex, labelset[g])]
        count += 1
        if 0 <= num_samples <= count:
            break
#        if idx > num_samples:
#            break
    data = []
    if num_samples > 1:
        print("num_samples", num_samples)
        print("By_Class[1]:", len(by_class[1]))
        print("By_Class[0]:", len(by_class[0]))
    if 1 in by_class:
        max_len = min(25000, len(by_class[1]))
        data.extend(by_class[1][:max_len])
        if num_samples > 1:
            data.extend(by_class[0][:max_len])
    else:
        max_len = 1
        data.extend(by_class[0][:max_len])
    return data

'''
def get_celeba(feature_idx, split_percentage=.8,
               num_train=np.float('inf'), num_test=np.float('inf')):
#    celeba_path = '~/datasets/'
    celeba_path = get_dataset_path()

    SIZE = 96
    transform = transforms.Compose(
        [transforms.Resize([SIZE,SIZE]),
         transforms.ToTensor()
        ])

    trainset = torchvision.datasets.CelebA(root=celeba_path,
                                           split='train',
                                           transform=transform,
                                           download=True)
    trainset = celeba_subset(trainset, feature_idx, num_samples=num_train)
    trainset, valset = split(trainset, p=split_percentage)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                              shuffle=True, num_workers=1)

    valloader = torch.utils.data.DataLoader(valset, batch_size=128,
                                            shuffle=False, num_workers=1)

    testset = torchvision.datasets.CelebA(root=celeba_path,
                                              split='test',
                                              transform=transform,
                                              download=True)
    testset = celeba_subset(testset, feature_idx, num_samples=num_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                             shuffle=False, num_workers=1)

    print("Train Size: ", len(trainset), "Val Size: ", len(valset), "Test Size: ", len(testset))
    return trainloader, valloader, testloader
'''

# Step 2: Function to prepare data loaders using official partition
def get_celeba(feature_idx, batch_size=128, num_train=float('inf'), num_test=float('inf'), num_val=float('inf')):
    SIZE = 96
    transform = transforms.Compose([
        transforms.Resize([SIZE, SIZE]),
        transforms.ToTensor()
    ])

    root = get_dataset_path()

#    root = os.path.join(root, 'celeba')

    # Load CelebA dataset (images + attributes)
    print('Files present:', os.listdir(root)[:10])

    celeba = CelebA_NoHash(root=root,split='all', target_type='attr',
                       transform=transform, download=False)

#    celeba = torchvision.datasets.CelebA(root=root, split='all', target_type='attr',
#                             transform=transform, download=False)

    # Read partition file to split data
    split_file = os.path.join(root,'celeba', 'list_eval_partition.txt')
    df = pd.read_csv(split_file, delim_whitespace=True, header=None, names=['image_id', 'partition'])

    # Get indices for train/val/test based on partition labels
    train_idx = df[df['partition'] == 0].index.tolist()
    val_idx   = df[df['partition'] == 1].index.tolist()
    test_idx  = df[df['partition'] == 2].index.tolist()

    # Create subsets for each split using selected attribute
    trainset = celeba_subset(celeba, feature_idx, train_idx,num_train)
    valset   = celeba_subset(celeba, feature_idx, val_idx, num_val)
    testset  = celeba_subset(celeba, feature_idx, test_idx, num_test)

    # Create PyTorch DataLoaders
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    valloader   = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=2)
    testloader  = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    print("Num Train: ", len(trainset), "Num Val: ", len(valset),
              "Num Test: ", len(testset))

    return trainloader, valloader, testloader


def group_by_class(dataset):
    labelset = {}
    for i in range(10):
        labelset[i] = []
    for i, batch in enumerate(dataset):
        img, label = batch
        labelset[label].append(img.view(1, 3, 32, 32))
    return labelset


def merge_data(cifar, mnist, n):
    cifar_by_label = group_by_class(cifar)
    mnist_by_label = group_by_class(mnist)

    merged_data = []
    merged_labels = []

    labelset = {}

    for i in range(10):
        one_hot = torch.zeros(1, 10)
        one_hot[0, i] = 1
        labelset[i] = one_hot

    for l in cifar_by_label:

        cifar_data = torch.cat(cifar_by_label[l])
        mnist_data = torch.cat(mnist_by_label[l])
        min_len = min(len(mnist_data), len(cifar_data))
        m = min(n, min_len)
        cifar_data = cifar_data[:m]
        mnist_data = mnist_data[:m]

        merged = torch.cat([cifar_data, mnist_data], axis=-1)
        #for i in range(3):
        #    vis.image(merged[i])
        merged_data.append(merged.reshape(m, -1))
        #print(merged.shape)
        merged_labels.append(np.repeat(labelset[l], m, axis=0))
    merged_data = torch.cat(merged_data, axis=0)

    merged_labels = np.concatenate(merged_labels, axis=0)
    merged_labels = torch.from_numpy(merged_labels)

    return list(zip(merged_data, merged_labels))

# For pytorch1.13.2
# def get_mnist(split_percentage=.8, num_train_per_class=np.float('inf'),
#               num_test_per_class=np.float('inf')):

# For ker_rfm
def get_mnist(split_percentage=.8, num_train_per_class=np.float64('inf'),
              num_test_per_class=np.float64('inf')):
    path = get_dataset_path()
    transform = transforms.Compose(
            [#transforms.Resize([32,32]),
                transforms.ToTensor()
            ])

    mnist_transform = transforms.Compose(
            [transforms.Resize([32,32]),
             transforms.ToTensor(),
             transforms.Lambda(lambda x: x.repeat(3, 1, 1))
            ])

    mnist_trainset = torchvision.datasets.MNIST(root=path,
                                                    train=True,
                                                    transform=mnist_transform,
                                                    download=True)

    trainset, valset = split(mnist_trainset, p=split_percentage)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                                  shuffle=True, num_workers=1)
    valloader = torch.utils.data.DataLoader(valset, batch_size=64,
                                                shuffle=False, num_workers=1)
    mnist_testset = torchvision.datasets.MNIST(root=path,
                                                   train=False,
                                                   transform=mnist_transform,
                                                   download=True)
    testloader = torch.utils.data.DataLoader(mnist_testset, batch_size=64,
                                                 shuffle=False, num_workers=1)

    print("Num Train: ", len(trainset), "Num Val: ", len(valset),
              "Num Test: ", len(testset))
    return trainloader, valloader, testloader

# For pytorch1.13.2
# def get_cifar_mnist(split_percentage=.8, num_train_per_class=np.float('inf'),
#                     num_test_per_class=np.float('inf')):
    
# For ker_rfm
def get_cifar_mnist(split_percentage=.8, num_train_per_class=np.float64('inf'),
                    num_test_per_class=np.float64('inf')):
#   path = '~/datasets/'
    path = get_dataset_path()

    transform = transforms.Compose(
            [#transforms.Resize([32,32]),
                transforms.ToTensor()
            ])

    mnist_transform = transforms.Compose(
            [transforms.Resize([32,32]),
             transforms.ToTensor(),
             transforms.Lambda(lambda x: x.repeat(3, 1, 1))
            ])


    cifar_trainset = torchvision.datasets.CIFAR10(root=path,
                                                      train=True,
                                                      transform=transform,
                                                      download=False)

    mnist_trainset = torchvision.datasets.MNIST(root=path,
                                                    train=True,
                                                    transform=mnist_transform,
                                                    download=False)
    trainset = merge_data(cifar_trainset, mnist_trainset, num_train_per_class)
    trainset, valset = split(trainset, p=split_percentage)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                                  shuffle=True, num_workers=2)
    valloader = torch.utils.data.DataLoader(valset, batch_size=128,
                                                shuffle=False, num_workers=1)

    cifar_testset = torchvision.datasets.CIFAR10(root=path,
                                                     train=False,
                                                     transform=transform,
                                                     download=False)

    mnist_testset = torchvision.datasets.MNIST(root=path,
                                                   train=False,
                                                   transform=mnist_transform,
                                                   download=False)

    testset = merge_data(cifar_testset, mnist_testset, num_test_per_class)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                                 shuffle=False, num_workers=2)

    print("Num Train: ", len(trainset), "Num Val: ", len(valset),
              "Num Test: ", len(testset))
    return trainloader, valloader, testloader


def draw_star(ex, v, c=3):
    ex[:c, 5:6, 7:14] = v
    ex[:c, 4, 9:12] = v
    ex[:c, 3, 10] = v
    ex[:c, 6, 8:13] = v
    ex[:c, 7, 9:12] = v
    ex[:c, 8, 8:13] = v
    ex[:c, 9, 8:10] = v
    ex[:c, 9, 11:13] = v
    return ex


def one_hot_stl_toy(dataset, num_samples=-1):
    labelset = {}
    for i in range(2):
        one_hot = torch.zeros(2)
        one_hot[i] = 1
        labelset[i] = one_hot

    subset = [(ex, label) for idx, (ex, label) in enumerate(dataset) \
              if idx < num_samples and (label == 0 or label == 9)]

    adjusted = []
    for idx, (ex, label) in enumerate(subset):
        if label == 9:
            ex = draw_star(ex,1, c=2)
            y = 1
        else:
            ex = draw_star(ex, 0)
            y = 0
        ex = ex.flatten()
        adjusted.append((ex, labelset[y]))
    return adjusted


# For pytorch1.13.2
# def get_stl_star(split_percentage=.8, num_train=np.float('inf'),
#                  num_test=np.float('inf')):

# For ker_rfm
def get_stl_star(split_percentage=.8, num_train=np.float64('inf'),
                 num_test=np.float64('inf')):
    SIZE = 96
    transform = transforms.Compose(
        [transforms.Resize([SIZE, SIZE]),
         transforms.ToTensor()
        ])

#    path = '~/datasets/'
    path = get_dataset_path()

    trainset = torchvision.datasets.STL10(root=path,
                                          split='train',
                                          #train=True,
                                          transform=transform,
                                          download=False)
    trainset = one_hot_stl_toy(trainset, num_samples=num_train)
    trainset, valset = split(trainset, p=split_percentage)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                              shuffle=True, num_workers=2)

    valloader = torch.utils.data.DataLoader(valset, batch_size=128,
                                            shuffle=False, num_workers=1)
    testset = torchvision.datasets.STL10(root=path,
                                         split='test',
                                         transform=transform,
                                         download=False)
    testset = one_hot_stl_toy(testset, num_samples=num_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                             shuffle=False, num_workers=2)
    print("Num Train: ", len(trainset), "Num Val: ", len(valset),
          "Num Test: ", len(testset))
    return trainloader, valloader, testloader
