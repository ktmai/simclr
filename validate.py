"""Load trained model and test it on a linear classifier
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from models import Encoder, Classifier
from discriminator_resnet import DiscriminatorNet


def train(args, model, device, train_loader, optimizer, epoch, criterion):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model.base.supervised_classification(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch + 1,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )


def validate(args, model, device, val_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in val_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model.base.supervised_classification(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(
        "Accuracy of the network on the 1000 val images: %d %%"
        % (100 * correct / total)
    )


def main():

    parser = argparse.ArgumentParser(description="Test SimCLR model")
    parser.add_argument(
        "--EPOCHS", default=1, type=int, help="Number of epochs for training"
    )
    parser.add_argument("--BATCH_SIZE", default=64, type=int, help="Batch size")
    args = parser.parse_args()

    from data_utils import train_loader_func, val_loader_func

    val_loader = val_loader_func()
    train_loader = train_loader_func()

    use_cuda = torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")

    pretrained_model = DiscriminatorNet()
    pretrained_model.load_state_dict(torch.load("trained_discriminator.pt"))
    # Freeze weights in the pretrained model
    for param in pretrained_model.parameters():
        param.requires_grad = False

    # Unfreeze supervised classifier on the end
    for param in pretrained_model.base.supervised_linear.parameters():
        param.requires_grad = True
    pretrained_model = pretrained_model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer_saved = optim.Adam(pretrained_model.base.supervised_linear.parameters())

    for epoch in range(20):
        print("Performance on the saved model")
        train(
            args,
            pretrained_model,
            device,
            train_loader,
            optimizer_saved,
            epoch,
            criterion,
        )
        validate(args, pretrained_model, device, val_loader)


if __name__ == "__main__":
    main()
