import pickle
import tarfile
from pathlib import Path

import numpy as np


CIFAR_MEAN = np.array((0.4914, 0.4822, 0.4465), dtype=np.float32).reshape(3, 1, 1)
CIFAR_STD = np.array((0.2023, 0.1994, 0.2010), dtype=np.float32).reshape(3, 1, 1)


class NumpyCIFARDataset:
    def __init__(self, data, labels, indices=None, augment=False, seed=42):
        self.data = data if indices is None else data[indices]
        self.labels = labels if indices is None else labels[indices]
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.labels)

    def get_batch(self, batch_indices):
        images = self.data[batch_indices].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
        if self.augment:
            images = self._augment_batch(images)
        images = (images - CIFAR_MEAN) / CIFAR_STD
        return images, self.labels[batch_indices]

    def _augment_batch(self, images):
        padded = np.pad(images, ((0, 0), (0, 0), (4, 4), (4, 4)), mode="constant")
        augmented = np.empty_like(images)

        for i in range(images.shape[0]):
            top = self.rng.integers(0, 9)
            left = self.rng.integers(0, 9)
            crop = padded[i, :, top : top + 32, left : left + 32]
            if self.rng.random() < 0.5:
                crop = crop[:, :, ::-1]
            augmented[i] = crop

        return augmented


class NumpyDataLoader:
    def __init__(self, dataset, batch_size=128, shuffle=False, seed=42):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        indices = np.arange(len(self.dataset))
        if self.shuffle:
            self.rng.shuffle(indices)

        for start in range(0, len(indices), self.batch_size):
            yield self.dataset.get_batch(indices[start : start + self.batch_size])


def _load_pickle_from_tar(tar_path, member_suffix):
    with tarfile.open(tar_path, "r:gz") as tar:
        member = next(m for m in tar.getmembers() if m.name.endswith(member_suffix))
        with tar.extractfile(member) as f:
            return pickle.load(f, encoding="latin1")


def _load_cifar10(root):
    tar_path = Path(root) / "cifar-10-python.tar.gz"
    train_batches = [_load_pickle_from_tar(tar_path, f"data_batch_{i}") for i in range(1, 6)]
    test_batch = _load_pickle_from_tar(tar_path, "test_batch")

    train_data = np.concatenate([batch["data"] for batch in train_batches], axis=0)
    train_labels = np.array(
        [label for batch in train_batches for label in batch["labels"]],
        dtype=np.int64,
    )
    test_data = test_batch["data"]
    test_labels = np.array(test_batch["labels"], dtype=np.int64)
    return train_data, train_labels, test_data, test_labels, 10


def _load_cifar100(root):
    tar_path = Path(root) / "cifar-100-python.tar.gz"
    train_batch = _load_pickle_from_tar(tar_path, "train")
    test_batch = _load_pickle_from_tar(tar_path, "test")

    train_data = train_batch["data"]
    train_labels = np.array(train_batch["fine_labels"], dtype=np.int64)
    test_data = test_batch["data"]
    test_labels = np.array(test_batch["fine_labels"], dtype=np.int64)
    return train_data, train_labels, test_data, test_labels, 100


def get_numpy_dataloaders(
    dataset_name="cifar10",
    batch_size=128,
    val_ratio=0.1,
    use_augmentation=True,
    seed=42,
    root="./data",
):
    if dataset_name == "cifar10":
        train_data, train_labels, test_data, test_labels, num_classes = _load_cifar10(root)
    elif dataset_name == "cifar100":
        train_data, train_labels, test_data, test_labels, num_classes = _load_cifar100(root)
    else:
        raise ValueError("dataset_name must be 'cifar10' or 'cifar100'")

    total_size = len(train_labels)
    val_size = int(total_size * val_ratio)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(total_size)
    val_indices = shuffled[:val_size]
    train_indices = shuffled[val_size:]

    train_set = NumpyCIFARDataset(train_data, train_labels, train_indices, augment=use_augmentation, seed=seed)
    val_set = NumpyCIFARDataset(train_data, train_labels, val_indices, augment=False, seed=seed)
    test_set = NumpyCIFARDataset(test_data, test_labels, augment=False, seed=seed)

    train_loader = NumpyDataLoader(train_set, batch_size=batch_size, shuffle=True, seed=seed)
    val_loader = NumpyDataLoader(val_set, batch_size=batch_size, shuffle=False, seed=seed)
    test_loader = NumpyDataLoader(test_set, batch_size=batch_size, shuffle=False, seed=seed)
    return train_loader, val_loader, test_loader, num_classes
