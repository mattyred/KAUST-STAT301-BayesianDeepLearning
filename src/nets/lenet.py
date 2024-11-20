import torch
from torch import nn

class LeNet5(nn.Module):
    def __init__(self, in_channels, output_dim, padding=0, dropout_rate=0.2):
        super(LeNet5, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 6, kernel_size=5, stride=1, padding=padding) # (6, 116, 116)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # (6, 58, 58)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=padding) # (16, 54, 54)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # (16, 27, 27)
        flattened_size = 16 * 27 * 27  
        self.fc1 = nn.Linear(flattened_size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                m.reset_parameters()
