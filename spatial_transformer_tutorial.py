# License: BSD
# Author: Ghassen Hamrouni

from __future__ import print_function
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import argparse
from torch.optim.lr_scheduler import StepLR
import torchvision.models as models
from transformer import Net
from discriminator import DiscriminatorNet
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






model = Net().to(device)



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
            color_params = model.colorization(data)

            try:
                MSE = torch.sum((identity_tensor - transformer_matrix) ** 2)
                color_MSE = torch.sum((color_params) ** 2)
            except:
                color_MSE = torch.sum((color_params[0:16]) ** 2)
                MSE = torch.sum((identity_tensor[0:16] - transformer_matrix) ** 2)

            perception_loss = 0.2 / (1 + MSE ** 2)
            color_loss = 0.5 / (1 + color_MSE ** 2)
            # print("perception loss", perception_loss)

            loss = color_loss + perception_loss - discriminator_loss
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
        cpu_model = Net()
        cpu_model.load_state_dict(torch.load("temp_model.pt"))

        data = next(iter(train_loader))[0]
        input_tensor = data.cpu()
        transformed_input_tensor = cpu_model.transform(data).cpu()

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


for epoch in range(1, 100):
    print("epoch", epoch)
    train(epoch)
    # Visualize the STN transformation on some input batch
    torch.save(model.state_dict(), "temp_model.pt")
    visualize_stn()
    
    plt.ioff()
    plt.savefig(str(epoch) + "_example.png")
    plt.close()
