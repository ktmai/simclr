from torchvision import datasets, transforms
import torch
from torch.utils.data import DataLoader, sampler

# https://github.com/pytorch/vision/issues/168
class ChunkSampler(sampler.Sampler):
    """Samples elements sequentially from some offset.
    Arguments:
        num_samples: # of desired datapoints
        start: offset where we should start selecting from
    """

    def __init__(self, num_samples, start=0):
        self.num_samples = num_samples
        self.start = start

    def __iter__(self):
        return iter(range(self.start, self.start + self.num_samples))

    def __len__(self):
        return self.num_samples


def train_loader_func(NUM_TRAIN=49000, NUM_VAL=1000):
    cifar10_train = datasets.CIFAR10(
        root=".",
        train=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    )

    train_loader_func = DataLoader(
        cifar10_train, batch_size=64, sampler=ChunkSampler(NUM_TRAIN)
    )

    return train_loader_func


def val_loader_func(NUM_TRAIN=49000, NUM_VAL=1000):
    cifar10_val = datasets.CIFAR10(
        root=".",
        train=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    )

    val_loader_func = DataLoader(
        cifar10_val, batch_size=64, sampler=ChunkSampler(NUM_VAL, NUM_TRAIN)
    )
    return val_loader_func


def mnist_train_loader_func(NUM_TRAIN=49000, NUM_VAL=1000):
    mnist_train = datasets.MNIST(
        root=".",
        train=True,
        transform=transforms.Compose(
            [transforms.ToTensor()]
        ),
    )

    mnist_train_loader_func = DataLoader(
        mnist_train, batch_size=64, sampler=ChunkSampler(NUM_TRAIN)
    )

    return mnist_train_loader_func


def mnist_val_loader_func(NUM_TRAIN=49000, NUM_VAL=1000):
    mnist_val = datasets.MNIST(
        root=".",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    )

    mnist_val_loader_func = DataLoader(
        mnist_val, batch_size=64, sampler=ChunkSampler(NUM_VAL, NUM_TRAIN)
    )
    return mnist_val_loader_func
