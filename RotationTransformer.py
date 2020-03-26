import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
from torchvision.transforms.functional import rotate

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
        rotation_logits = F.softmax(x)
        rotation_factor = torch.tensor(torch.max(rotation_logits, dim=1)[1])
        augmented = []
        not_augmented = []
        augmented_targets = []
        not_augmented_targets = []
        for index, image in enumerate(input_images):
            if rotation_factor[index] != 0:
                augmented.append(rotate(image, rotation_factor[index]*15))
                augmented_targets.append(1)
            else:
                not_augmented.append(kornia.rotate(image, rotation_factor[index]*15))
                not_augmented_targets.append(rotation_factor[index])
        augmented = torch.stack(augmented)
        not_augmented = torch.stack(not_augmented)
        augmented_targets = torch.tensor(augmented_targets, dtype=torch.float)
        not_augmented_targets = torch.tensor(not_augmented_targets, dtype=torch.float)
        return augmented, augmented_targets, not_augmented, not_augmented_targets
