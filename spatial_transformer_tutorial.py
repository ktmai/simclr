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
from data_utils import train_loader_func
from utils import visualize_stn


def train(epoch, train_loader, device, discriminator, transformer,
          transformer_opt, discriminator_opt):
    identity_tensor = torch.load("identity_theta.pt")
    identity_tensor = identity_tensor.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    discriminator.train()
    transformer.train()
    discriminator_step = True
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        augmented = transformer.transform(data)
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
            transformer_opt.zero_grad()
            transformer_matrix = transformer.stn(data)[1]
            color_params = transformer.colorization(data)

            try:
                MSE = torch.sum((identity_tensor - transformer_matrix) ** 2)
                color_MSE = torch.sum((color_params) ** 2)
            except BaseException:
                color_MSE = torch.sum((color_params[0:16]) ** 2)
                MSE = torch.sum(
                    (identity_tensor[0: 16] - transformer_matrix) ** 2)

            perception_loss = 1 / (1 + MSE ** 2)
            color_loss = 1 / (1 + color_MSE ** 2)
            # print("perception loss", perception_loss)

            loss = color_loss + perception_loss - discriminator_loss
            loss.backward()
            transformer_opt.step()
            discriminator_step = True


def main():
    plt.ion()  # interactive mode

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transformer = Transformer().to(device)

    transformer_opt = optim.SGD(transformer.parameters(), lr=0.01)
    discriminator = DiscriminatorNet().to(device)
    discriminator_opt = optim.SGD(discriminator.parameters(), lr=0.01)
    train_loader = train_loader_func()
    for epoch in range(1, 5):
        print("epoch", epoch)
        train(epoch, train_loader, device, discriminator, transformer, transformer_opt, discriminator_opt)
        # Visualize the STN transformation on some input batch
        torch.save(transformer.state_dict(), "temp_transformer.pt")
        visualize_stn(
            train_loader=train_loader,
            temp_model_path="temp_transformer.pt")

        plt.ioff()
        plt.savefig(str(epoch) + "_example.png")
        plt.close()

    torch.save(discriminator.state_dict(), "trained_discriminator.pt")
    torch.save(transformer.state_dict(), "trained_transformer.pt")

if __name__ == "__main__":
    main()
