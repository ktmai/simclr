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
from models import Encoder
from pytorch_metric_learning import losses


def train(
    epoch,
    train_loader,
    device,
    discriminator,
    transformer,
    model,
    transformer_opt,
    discriminator_opt,
    model_opt,
    loss_func,
):
    criterion = torch.nn.BCELoss()
    model.train()
    discriminator.train()
    transformer.train()

    for batch_idx, (xi, _) in enumerate(train_loader):
        xi = xi.to(device)
        xj = transformer(xi)
        xj = xj.to(device)
        
        # discriminator_scores = discriminator(xj)
        discriminator_opt.zero_grad()
        discriminator_loss = (
            criterion(discriminator(xj), torch.ones(xj.size()[0]).to(device))
            + criterion(discriminator(xi), torch.zeros(xi.size()[0]).to(device))
        ) / 2
        discriminator_loss.backward(retain_graph=True) # TODO hack need to understand
        discriminator_opt.step()

        
        discriminator_scores = discriminator(xj)

        if torch.mean(discriminator_scores) < 0.5:
            
            hi = model(xi)
            hj = model(xj)
            embeddings = torch.cat((hi, hj))

            indices = torch.arange(0, 64).to(device)
            labels = torch.cat((indices, indices))
            loss = loss_func(embeddings, labels)
            transformer_loss = -loss

            model_opt.zero_grad()
            loss.backward(retain_graph=True)
            model_opt.step()
        
            transformer_opt.zero_grad()
            transformer_loss += criterion(
                discriminator(xj), torch.zeros(xj.size()[0]).to(device)
            )
            transformer_loss.backward()
            transformer_opt.step()
            continue

        transformer_opt.zero_grad()
        transformer_loss = criterion(
            discriminator(xj), torch.zeros(xj.size()[0]).to(device)
        )
        transformer_loss.backward()
        transformer_opt.step()


        if batch_idx % 10 == 0:
            break
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.10f}".format(
                    epoch + 1,
                    batch_idx * len(xi),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    transformer_loss.item(),
                )
            )


def main():
    from rotation_classifier import RotationClassifier
    from transformer import Transformer
    from models import Encoder

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loss_func = losses.NTXentLoss(0.5)

    transformer = Transformer().to(device)
    discriminator = RotationClassifier().to(device)
    model = Encoder().to(device)
    model_opt = optim.Adam(model.parameters())

    transformer_opt = optim.Adam(transformer.parameters(), lr=0.01)
    discriminator_opt = optim.SGD(discriminator.parameters(), lr=0.01)
    train_loader = train_loader_func()

    for epoch in range(1, 200):
        print("epoch", epoch)
        train(
            epoch=epoch,
            train_loader=train_loader,
            device=device,
            discriminator=discriminator,
            transformer=transformer,
            model=model,
            transformer_opt=transformer_opt,
            discriminator_opt=discriminator_opt,
            model_opt=model_opt,
            loss_func=loss_func
        )
        # Visualize the STN transformation on some input batch
        torch.save(transformer.state_dict(), "temp_transformer.pt")
        visualize_stn(
            train_loader=train_loader, temp_model_path="temp_transformer.pt"
        )

        plt.ioff()
        plt.savefig(str(epoch) + "_example.png")
        plt.close()

    torch.save(discriminator.state_dict(), "trained_discriminator.pt")
    torch.save(transformer.state_dict(), "trained_transformer.pt")
    torch.save(model.state_dict(), "trained_encoder.pt")


if __name__ == "__main__":
    main()
