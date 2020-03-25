import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
import kornia


class RotationTransformer(nn.Module):
    def __init__(self):
        super(RotationTransformer, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 8)

    def forward(self, x):
        input_images = x.clone()
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = F.log_softmax(x)
        rotation_factor = torch.tensor(torch.max(x, dim=1)[1])
        rotated_images = []
        for index, image in enumerate(input_images):
            rotated_images.append(kornia.rotate(image, rotation_factor[index]*15))
        x = torch.stack(rotated_images)
        return x, rotation_factor
