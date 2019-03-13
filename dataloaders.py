"""Data loading utilities

Intended use:
importing a get_data_loader(datasetname) from this module

Currently has the following datasets:
- MNIST
- CIFAR10

"""

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, CIFAR10

from constants import DEVICE, N_WORKERS


def get_data_loader(dataset_name, mode, batch_size=100):
    if dataset_name == "mnist":
        if mode == "train":
            transform_train = transforms.Compose([
              transforms.ToTensor(),
              transforms.Normalize((0.1307,), (0.3081,))
            ])
            train_data = MNIST(root='./data', train=True, download=True, transform=transform_train)
            train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=N_WORKERS)
            return train_loader

        elif mode in ("val", "dev"):
            transform_test = transforms.Compose([
              transforms.ToTensor(),
              transforms.Normalize((0.1307,), (0.3081,))
            ])
            val_data = MNIST(root='./data', train=False, download=True, transform=transform_test)
            val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=N_WORKERS)
            return val_loader

        elif mode == "test":
            raise ValueError("No need to use a test dataset.")

    elif dataset_name == "cifar10":
        if mode == "train":
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

            train_data = CIFAR10(root='./data', train=True, download=True, transform=transform_train)
            train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=N_WORKERS)
            return train_loader

        elif mode in ("val", "dev"):
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            val_data = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
            val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=N_WORKERS)
            return val_loader

        elif mode == "test":
            raise ValueError("No need to use a test dataset.")

        else:
            ValueError("Unknown mode {}".format(mode))
    else:
        raise ValueError("Unknown dataset {}".format(dataset_name))


