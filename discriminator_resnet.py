import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
from resnet import ResNet50


class DiscriminatorNet(nn.Module):
    def __init__(self, base_model=models.resnet50):
        super(DiscriminatorNet, self).__init__()
        self.base = ResNet50(num_classes=2)

    def forward(self, x):
        x = self.base(x)
        return x
