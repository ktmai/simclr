# License: BSD
# Author: Ghassen Hamrouni

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR


import torch
import torch.nn as nn
import torchvision.models as models

plt.ion()  # interactive mode

device = torch.device('cuda:1')

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
    batch_size=2,
    shuffle=True,
    num_workers=4,
)
# Test dataset

### help from https://github.com/aicaffeinelife/Pytorch-STN/blob/master/models/STNModule.py
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.MaxPool2d(2),
            nn.ReLU(True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.MaxPool2d(2),
            nn.ReLU(True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.MaxPool2d(2),
            nn.ReLU(True),
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(32 * 4 * 4, 1024), nn.ReLU(True), nn.Linear(1024, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(
            torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float)
        )

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 32 * 4 * 4)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        return x, theta

    def transform_with_theta(self, x, theta):
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x

    def transform(self, x):
        x, theta = self.stn(x)
        x = self.transform_with_theta(x, theta)
        return x

    def forward(self, x):
        # transform the input
        x = self.transform(x)

        return F.log_softmax(x, dim=1)


model = Net().to(device)


class DiscriminatorNet(nn.Module):
    def __init__(self, base_model=models.resnet50):
        super(DiscriminatorNet, self).__init__()

        self.base = []

        # Amend Resnet-50
        for name, module in base_model().named_children():
            if name == "conv1":
                module = nn.Conv2d(
                    3, 64, kernel_size=3, stride=1, padding=1, bias=False
                )
            if not isinstance(module, nn.Linear) and not isinstance(
                module, nn.MaxPool2d
            ):
                self.base.append(module)
        self.base = nn.Sequential(*self.base)
        self.g = nn.Sequential(
            nn.Linear(2048, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.base(x)
        x = torch.flatten(x, start_dim=1)
        out = self.g(x)
        return out


optimizer = optim.SGD(model.parameters(), lr=0.01)
discriminator = DiscriminatorNet().to(device)
discriminator_opt = optim.SGD(discriminator.parameters(), lr=0.01)


def train(epoch):
    import torch

    identity_tensor = torch.load("identity_theta.pt")
    identity_tensor = identity_tensor.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    discriminator.train()
    model.train()
    discriminator_step = True
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        # optimizer.zero_grad()
        augmented = model.transform(data)
        augmented = augmented.to(device)
        true_targets = torch.ones(target.size()).long()
        false_targets = torch.zeros(target.size()).long()
        true_targets = true_targets.to(device)
        false_targets = false_targets.to(device)
        discriminator_out_true = discriminator(data)
        discriminator_out_false = discriminator(augmented)
        discriminator_loss = criterion(
            discriminator_out_true, true_targets
        ) + criterion(discriminator_out_false, false_targets)

        if discriminator_step:
            discriminator_opt.zero_grad()
            discriminator_loss.backward()
            discriminator_opt.step()
            discriminator_step = False
        else:
            optimizer.zero_grad()
            transformer_matrix = model.stn(data)[1]
            try:
                MSE = torch.sum((identity_tensor - transformer_matrix) ** 2)

            except:
                MSE = torch.sum((identity_tensor[0:32] - transformer_matrix) ** 2)

            perception_loss = 0.2 / (1 + MSE ** 2)
            print("perception loss", perception_loss)
            loss = perception_loss - discriminator_loss
            loss.backward()
            optimizer.step()
            discriminator_step = True

def convert_image_np(inp):
    """Convert a Tensor to numpy image."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    return inp


def visualize_stn():
    with torch.no_grad():
        # Get a batch of training data
        data = next(iter(train_loader))[0].to(device)

        input_tensor = data.cpu()
        transformed_input_tensor = model.transform(data).cpu()

        in_grid = convert_image_np(torchvision.utils.make_grid(input_tensor))

        out_grid = convert_image_np(
            torchvision.utils.make_grid(transformed_input_tensor)
        )

        # Plot the results side-by-side
        f, axarr = plt.subplots(1, 2)
        axarr[0].imshow(in_grid)
        axarr[0].set_title("Dataset Images")

        axarr[1].imshow(out_grid)
        axarr[1].set_title("Transformed Images")


for epoch in range(1, 20 + 1):
    train(epoch)
    # Visualize the STN transformation on some input batch
    visualize_stn()

    plt.ioff()
    plt.savefig(str(epoch) + "_example.png")
    plt.close()
