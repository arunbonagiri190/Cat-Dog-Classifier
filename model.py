import torch
import torch.nn as nn
import torch.nn.functional as F

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=5, stride=4)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(576, 2)
        
        self.bc1 = nn.BatchNorm2d(16)
        self.bc2 = nn.BatchNorm2d(64)
        self.do1 = nn.Dropout(0.2)
        
    def forward(self, x):
        
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = self.bc1(F.max_pool2d(F.relu(self.conv2(x)), 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        x = self.bc2(F.max_pool2d(F.relu(self.conv4(x)), 2))
        x = torch.flatten(x, start_dim=-3)
        x = self.do1(self.fc1(x))
        
        return x