"""Implementation of training loop.

Some modifications to the original paper:
    Adam optimiser instead of LARS
    Trained on CIFAR-10 instead of ImageNet
"""

import argparse
import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from pytorch_metric_learning import losses
from data_augmentations import get_color_distortion
from models import Encoder
from modified_cifar import CIFAR10_new


def train(args, model, device, train_loader, optimizer, loss_func, epoch):
    label_loss = nn.CrossEntropyLoss() 
    model.train()
    for batch_idx, (xi, xj, _, true_Ti, true_Tj) in enumerate(train_loader):
        xi, xj = xi.to(device), xj.to(device)
        true_Ti, true_Tj = true_Ti.to(device), true_Tj.to(device)
        optimizer.zero_grad()
        hi, predicted_Ti = model(xi)
        hj, predicted_Tj = model(xj)
        embeddings = torch.cat((hi, hj))
        # Create fake labels for each sample
        indices = torch.arange(0, args.BATCH_SIZE).to(device)
        labels = torch.cat((indices, indices))
        loss = loss_func(embeddings, labels)
        loss += label_loss(predicted_Ti, true_Ti)/2
        loss += label_loss(predicted_Tj, true_Tj)/2
        loss.backward()
        optimizer.step()
        if batch_idx % args.LOG_INT == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.10f}".format(
                    epoch + 1,
                    batch_idx * len(xi),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )


def main():
    parser = argparse.ArgumentParser(description="Implementation of SimCLR")
    parser.add_argument(
        "--EPOCHS", default=10, type=int, help="Number of epochs for training"
    )
    parser.add_argument("--BATCH_SIZE", default=64, type=int, help="Batch size")
    parser.add_argument(
        "--TEMP", default=0.5, type=float, help="Temperature parameter for NT-Xent"
    )
    parser.add_argument(
        "--LOG_INT",
        default=100,
        type=int,
        help="How many batches to wait before logging training status",
    )
    parser.add_argument(
        "--DISTORT_STRENGTH",
        default=0.5,
        type=float,
        help="Strength of colour distortion",
    )
    parser.add_argument("--SAVE_NAME", default="model")
    args = parser.parse_args()
    use_cuda = torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")

    online_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop((32, 32)),
            transforms.RandomHorizontalFlip(),
            get_color_distortion(s=args.DISTORT_STRENGTH),
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
        ]
    )

    trainset = CIFAR10_new(
        root="./data", train=True, download=True, transform=online_transform
    )

    # Need to drop last minibatch to prevent matrix multiplication erros
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=args.BATCH_SIZE, shuffle=True, drop_last=True
    )

    model = Encoder().to(device)
    optimizer = optim.Adam(model.parameters())
    loss_func = losses.NTXentLoss(args.TEMP)
    for epoch in range(args.EPOCHS):
        train(args, model, device, train_loader, optimizer, loss_func, epoch)

    torch.save(model.state_dict(), "./ckpt/{}.pth".format(args.SAVE_NAME))


if __name__ == "__main__":
    main()
