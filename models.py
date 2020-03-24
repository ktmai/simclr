"""Models used for training and testing SimCLR

Includes modified Resnet-50 encoder and linear classifier which includes all
layers of Resnet-50 except for the projection head
"""

import torch
import torch.nn as nn
import torchvision.models as models


class Encoder(nn.Module):
    def __init__(self, base_model=models.resnet50):
        super(Encoder, self).__init__()

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
        self.g = nn.Sequential(nn.Linear(2048, 128), nn.BatchNorm1d(128), nn.ReLU())

    def forward(self, x):
        x = self.base(x)
        x = torch.flatten(x, start_dim=1)
        out = self.g(x)
        return out


class Classifier(nn.Module):
    def __init__(self, old_net):
        super(Classifier, self).__init__()
        self.base = old_net.base
        self.fc = nn.Linear(2048, 10)

    def forward(self, x):
        x = self.base(x)
        x = torch.flatten(x, start_dim=1)
        out = self.fc(x)
        return out
