from torchvision import datasets, transforms
import torch
from torch.utils.data import DataLoader, sampler

# Training dataset
train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10(
        root=".",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    ),
    batch_size=64,
    shuffle=True,
    num_workers=4,
)

# https://github.com/pytorch/vision/issues/168
class ChunkSampler(sampler.Sampler):
    """Samples elements sequentially from some offset. 
    Arguments:
        num_samples: # of desired datapoints
        start: offset where we should start selecting from
    """
    def __init__(self, num_samples, start = 0):
        self.num_samples = num_samples
        self.start = start

    def __iter__(self):
        return iter(range(self.start, self.start + self.num_samples))

    def __len__(self):
        return self.num_samples

NUM_TRAIN = 49000
NUM_VAL = 1000

cifar10_train = datasets.CIFAR10('./cs231n/datasets', train=True, download=True,
                           transform=torch.ToTensor())
loader_train = DataLoader(cifar10_train, batch_size=64, sampler=ChunkSampler(NUM_TRAIN, 0))

cifar10_val = datasets.CIFAR10('./cs231n/datasets', train=True, download=True,
                           transform=torch.ToTensor())
loader_val = DataLoader(cifar10_val, batch_size=64, sampler=ChunkSampler(NUM_VAL, NUM_TRAIN))
