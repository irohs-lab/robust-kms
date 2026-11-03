import torch
import dataset


def loader_to_tensors(loader, device, flatten=True):
    xs, ys = [], []

    for x, y in loader:
        if flatten and x.ndim > 2:
            x = x.view(x.size(0), -1)

        xs.append(x)
        ys.append(y)

    x = torch.cat(xs, dim=0).to(device)
    y = torch.cat(ys, dim=0).to(device)

    return x, y


def get_class_pairs(dataset_name, multiclass):
    dataset_name = dataset_name.lower()

    if multiclass or dataset_name in ["bank"]:
        return [0], [1]

    classes1 = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    classes2 = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    return classes1, classes2


def load_data_for_experiment(dataset_name, cl1, cl2, multiclass):
    dataset_name = dataset_name.lower()

    if dataset_name == "svhn":
        trainloader, valloader, testloader = dataset.get_svhn(classes=(cl1, cl2), multiclass=multiclass,)
        print("I am SVHN")

    elif dataset_name in ["fmnist", "qmnist", "kmnist", "emnist"]:
        trainloader, valloader, testloader = dataset.load_fmnist(
            classes=(cl1, cl2),
            which_mnist=dataset_name,
            multiclass=multiclass,
        )
        print(f"I am {dataset_name.upper()}")
    elif dataset_name == "cifar10":
        trainloader, valloader, testloader = dataset.get_cifar(classes =(c1,c2),multiclass = multiclass,)
        print(f"I am {dataset_name.upper()}")

    elif dataset_name == "bank":
        trainloader, testloader = dataset.uci_bank()
        valloader = testloader
        print("I am Bank")

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    return trainloader, valloader, testloader