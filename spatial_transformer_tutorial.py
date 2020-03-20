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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
# Test dataset
test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10(
        root=".",
        train=False,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    ),
    batch_size=64,
    shuffle=True,
    num_workers=4,
)

### help from https://github.com/aicaffeinelife/Pytorch-STN/blob/master/models/STNModule.py
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(3, 6, 5),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(400, 64), nn.ReLU(True), nn.Linear(64, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(
            torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float)
        )

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 16 * 5 * 5)
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

        # Perform the usual forward pass
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


model = Net().to(device)


class DiscriminatorNet(nn.Module):
    def __init__(self):
        super(DiscriminatorNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


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
                MSE = torch.sum((identity_tensor[0:16] - transformer_matrix) ** 2)

            perception_loss = 0.2 / (1 + MSE ** 2)
            print("perception loss", perception_loss)
            loss = perception_loss - discriminator_loss
            loss.backward()
            optimizer.step()
            discriminator_step = True


def test():
    with torch.no_grad():
        model.eval()
        test_loss = 0
        correct = 0
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            # sum up batch loss
            test_loss += F.nll_loss(output, target, size_average=False).item()
            # get the index of the max log-probability
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        print(
            "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
                test_loss,
                correct,
                len(test_loader.dataset),
                100.0 * correct / len(test_loader.dataset),
            )
        )


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
        data = next(iter(test_loader))[0].to(device)

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
