from torchvision import datasets, transforms
import torch

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
