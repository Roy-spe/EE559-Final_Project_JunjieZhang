from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import torch


def get_transforms(use_augmentation=True):
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)

    if use_augmentation:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    return train_transform, test_transform


def get_dataloaders(dataset_name="cifar10", batch_size=128, val_ratio=0.1,
                    num_workers=2, use_augmentation=True, seed=42):
    train_transform, test_transform = get_transforms(use_augmentation)

    if dataset_name == "cifar10":
        train_full_aug = datasets.CIFAR10("./data", train=True, download=True, transform=train_transform)
        train_full_plain = datasets.CIFAR10("./data", train=True, download=True, transform=test_transform)
        test_set = datasets.CIFAR10("./data", train=False, download=True, transform=test_transform)
        num_classes = 10

    elif dataset_name == "cifar100":
        train_full_aug = datasets.CIFAR100("./data", train=True, download=True, transform=train_transform)
        train_full_plain = datasets.CIFAR100("./data", train=True, download=True, transform=test_transform)
        test_set = datasets.CIFAR100("./data", train=False, download=True, transform=test_transform)
        num_classes = 100

    else:
        raise ValueError("dataset_name must be 'cifar10' or 'cifar100'")

    total_size = len(train_full_aug)
    val_size = int(total_size * val_ratio)
    train_size = total_size - val_size

    generator = torch.Generator().manual_seed(seed)
    train_split, val_split = random_split(range(total_size), [train_size, val_size], generator=generator)

    train_set = torch.utils.data.Subset(train_full_aug, train_split.indices)
    val_set = torch.utils.data.Subset(train_full_plain, val_split.indices)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, num_classes