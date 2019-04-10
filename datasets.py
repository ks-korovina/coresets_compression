"""Data loading utilities

Currently has the following datasets:
- MNIST
- CIFAR10

TODO:
* Do max-min normalization if possible
  (https://pytorch.org/docs/stable/torchvision/transforms.html)

"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, CIFAR10

from constants import DEVICE, N_WORKERS


def sample_from_dataset(dataset, size):
    """ Utility needed for corenet """
    inds = np.random.choice(len(dataset), size=size, replace=False)
    # make one batch, hopefully it will not be huge for start
    # (fix this later)
    # also TODO: nicer shape fixing
    examples = [dataset[i][0].flatten() for i in inds]
    return torch.stack(examples, dim=0)


def get_data_loader(dataset_name, is_train, batch_size=100):
    dataset = get_dataset(dataset_name, is_train)
    data_loader = DataLoader(dataset, batch_size=batch_size,
                             shuffle=is_train, num_workers=N_WORKERS)
    return data_loader


def get_dataset(dataset_name, is_train):
    if dataset_name == "mnist":
        return get_mnist(is_train)
    elif dataset_name == "cifar10":
        return get_cifar10(is_train)
    else:
        raise ValueError("Unknown dataset {}".format(dataset_name))


def get_debug():
    return DebugDataset()


def get_mnist(is_train):
    # for some reason, still loading even if already there
    do_download = True  #(not os.path.isdir("./data/MNIST/"))
    # print(os.getcwd()+'/data')
    if is_train:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.1307,), (0.3081,))
            transforms.Normalize((0.,), (0.3081,))
        ])
        train_data = MNIST(root='data',
                           train=True, download=do_download,
                           transform=transform_train)
        return train_data

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.1307,), (0.3081,))
        transforms.Normalize((0.,), (0.3081,))
    ])
    val_data = MNIST(root='data',
                    train=False, download=do_download,
                    transform=transform_test)
    return val_data


def get_cifar10(is_train):
    if is_train:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_data = CIFAR10(root='data', train=True, download=True,
                             transform=transform_train)
        return train_data

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    val_data = torchvision.datasets.CIFAR10(root='data', train=False, download=True,
                                            transform=transform_test)
    return val_data


class DebugDataset(Dataset):
    def __init__(self):
        self.xdata = torch.rand((100, 5))

    def __len__(self):
        return len(self.xdata)

    def __getitem__(self, idx):
        return self.xdata[idx], 0


if __name__ == "__main__":
    ds = get_mnist(False)
    print(ds.processed_folder)
    print(ds._check_exists())


