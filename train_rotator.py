# License: BSD
# Original Author: Ghassen Hamrouni altered now

from __future__ import print_function
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import argparse
from torch.optim.lr_scheduler import StepLR
from transformer import Transformer
from discriminator_resnet import DiscriminatorNet
from data_utils import mnist_train_loader_func
from utils import visualize_stn
from RotationTransformer import RotationTransformer
from rotation_classifier import RotationClassifier


def train(
    epoch,
    train_loader,
    device,
    discriminator,
    transformer,
    transformer_opt,
    discriminator_opt,
):
    criterion = torch.nn.BCELoss()
    discriminator.train()
    transformer.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, _ = data.to(device), target.to(device)
        augmented, augmented_target, not_augmented, not_augmented_target = transformer(
            data
        )
        augmented = augmented.to(device)

        transformer_opt.zero_grad()
        generator_loss = criterion(
            discriminator(augmented), torch.zeros(augmented_target.size())
        )

        generator_loss.backward()
        transformer_opt.step()

        discriminator_opt.zero_grad()
        discriminator_loss = (
            criterion(discriminator(augmented), augmented_target)
            + criterion(discriminator(not_augmented), not_augmented_target)
        ) / 2

        discriminator_loss.backward()
        discriminator_opt.step()

    return discriminator_loss, generator_loss


def main():
    plt.ion()  # interactive mode

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transformer = RotationTransformer().to(device)
    transformer_opt = optim.Adam(transformer.parameters(), lr=0.01)
    discriminator = RotationClassifier().to(device)
    discriminator_opt = optim.SGD(discriminator.parameters(), lr=0.01)
    train_loader = mnist_train_loader_func()
    disc_losses = []
    gen_losses = []
    for epoch in range(1, 20):
        print("epoch", epoch)
        discriminator_loss, generator_loss = train(
            epoch,
            train_loader,
            device,
            discriminator,
            transformer,
            transformer_opt,
            discriminator_opt,
        )
        disc_losses.append(discriminator_loss)
        gen_losses.append(generator_loss)
        # Visualize the STN transformation on some input batch
        torch.save(transformer.state_dict(), "temp_transformer.pt")
        # if epoch % 20 == 0:
        visualize_stn(train_loader=train_loader, temp_model_path="temp_transformer.pt")

        plt.ioff()
        plt.savefig(str(epoch) + "_example.png")
        plt.close()
    plt.plot(range(1, 20), disc_losses)
    plt.plot(range(1, 20), gen_losses)
    plt.savefig("discriminator_loss.png")
    torch.save(discriminator.state_dict(), "trained_discriminator.pt")
    torch.save(transformer.state_dict(), "trained_transformer.pt")


if __name__ == "__main__":
    main()
