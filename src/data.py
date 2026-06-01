"""CIFAR-10/100 data loaders with standard augmentation."""
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR100_STD = (0.2673, 0.2564, 0.2762)

_ROOT = None  # set by get_loaders


def _stats(dataset):
    return (CIFAR100_MEAN, CIFAR100_STD) if dataset == "cifar100" else (CIFAR10_MEAN, CIFAR10_STD)


def get_loaders(dataset="cifar100", root="./data", batch_size=128, num_workers=4, augment=True):
    mean, std = _stats(dataset)
    train_tf = transforms.Compose(
        ([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()] if augment else [])
        + [transforms.ToTensor(), transforms.Normalize(mean, std)]
    )
    test_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    DS = datasets.CIFAR100 if dataset == "cifar100" else datasets.CIFAR10
    train = DS(root, train=True, download=True, transform=train_tf)
    test = DS(root, train=False, download=True, transform=test_tf)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=False)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader


def num_classes(dataset):
    return 100 if dataset == "cifar100" else 10
