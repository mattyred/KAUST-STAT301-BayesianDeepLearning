import torch
from torch import nn

from ..layers.conv import Conv2d
from ..layers.linear import Linear

import torch.nn.functional as F

class LeNet5(nn.Module):
    def __init__(self, in_channels, output_dim, padding, scaled_variance=True, norm_layer=None, dropout_rate=0.2):
        super(LeNet5, self).__init__(),

        self.task = "classification"
                 
        self.feature = nn.Sequential(
            #1
            nn.Conv2d(in_channels=in_channels, out_channels=6*in_channels, kernel_size=5, padding=padding),   # 28*28->32*32-->28*28
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),  # 14*14
            
            #2
            nn.Conv2d(in_channels=6*in_channels, out_channels=16*in_channels, kernel_size=5),  # 10*10
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),  # 5*5
            
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            
            # Fully Connected Layer 1 + Dropout
            nn.Linear(in_features=16*5*5, out_features=120),
            nn.Tanh(),
            nn.Dropout(p=dropout_rate),
            
            # Fully Connected Layer 2 + Dropout
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh(),
            nn.Dropout(p=dropout_rate),
            
            # Output Layer
            nn.Linear(in_features=84, out_features=output_dim),
        )
        
    def forward(self, X, log_softmax=False):

        X = self.classifier(self.feature(X))
        
        if (self.task == "classification") and log_softmax:
            X = F.log_softmax(X, dim=1)

        return X


    def predict(self, X):

        self.eval()
        # if self.task == "classification":
        return self.forward(X, log_softmax=True)
        # else:
        #     return self.forward(X, log_softmax=False)
    

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, Linear) or isinstance(m, Conv2d):
                m.reset_parameters()
