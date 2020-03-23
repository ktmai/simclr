import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F


class DiscriminatorNet(nn.Module):
    
    def __init__(self, base_model = models.resnet50):
        super(DiscriminatorNet, self).__init__()
        
        self.base = []
        
        # Amend Resnet-50
        for name, module in base_model().named_children():
            if name == 'conv1':
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                self.base.append(module)
        self.base = nn.Sequential(*self.base)
        self.g = nn.Sequential(nn.Linear(2048, 128), nn.BatchNorm1d(128),
                               nn.ReLU(), nn.Linear(128, 2))
        
    def forward(self, x):
        x = self.base(x)
        x = torch.flatten(x, start_dim = 1)
        out = self.g(x)
        return out

#class DiscriminatorNet(nn.Module):
#    def __init__(self):
#        super(DiscriminatorNet, self).__init__()
#        self.conv1 = nn.Conv2d(3, 6, 5)
#        self.pool = nn.MaxPool2d(2, 2)
#        self.conv2 = nn.Conv2d(6, 16, 5)
#        self.fc1 = nn.Linear(16 * 5 * 5, 120)
#        self.fc2 = nn.Linear(120, 84)
#        self.fc3 = nn.Linear(84, 2)
#
#    def forward(self, x):
#        x = self.pool(F.relu(self.conv1(x)))
#        x = self.pool(F.relu(self.conv2(x)))
#        x = x.view(-1, 16 * 5 * 5)
#        x = F.relu(self.fc1(x))
#        x = F.relu(self.fc2(x))
#        x = self.fc3(x)
#        return x
