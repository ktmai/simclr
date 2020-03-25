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


def train(epoch, train_loader, device, discriminator, transformer,
          transformer_opt, discriminator_opt):
    criterion = torch.nn.CrossEntropyLoss()
    discriminator.train()
    transformer.train()
    discriminator_step = True

    for batch_idx, (data, target) in enumerate(train_loader):
        data, _ = data.to(device), target.to(device)
        augmented, target = transformer(data)
        augmented = augmented.to(device)
        discriminator_out = discriminator(augmented)
        discriminator_loss = criterion(discriminator_out, target)

        if discriminator_step:
            discriminator_opt.zero_grad()
            discriminator_loss.backward()
            discriminator_opt.step()
            discriminator_step = False
        else:
            transformer_opt.zero_grad()
            loss = - discriminator_loss
            loss.backward()
            transformer_opt.step()
            discriminator_step = True
        if batch_idx == 5:
            break

def main():
    plt.ion()  # interactive mode

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transformer = RotationTransformer().to(device)
    transformer_opt = optim.SGD(transformer.parameters(), lr=0.1)
    discriminator = RotationClassifier().to(device)
    discriminator_opt = optim.SGD(discriminator.parameters(), lr=0.1)
    train_loader = mnist_train_loader_func()

    for epoch in range(1, 200):
        print("epoch", epoch)
        train(epoch, train_loader, device, discriminator, transformer, transformer_opt, discriminator_opt)
        # Visualize the STN transformation on some input batch
        torch.save(transformer.state_dict(), "temp_transformer.pt")
        if epoch % 20 == 0:
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
