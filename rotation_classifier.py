import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

img_shape = (3, 32, 32)

# https://raw.githubusercontent.com/eriklindernoren/PyTorch-GAN/master/implementations/wgan_gp/wgan_gp.py
class RotationClassifier(nn.Module):
    def __init__(self):
        super(RotationClassifier, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        return validity
