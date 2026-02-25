import torch
import torch.nn as nn
import torch.nn.functional as F

## **********
## Definition of the model as a nn.Sequential
class MLP_shallow(nn.Module):
    def __init__(self, Din, Nneurons, Dout):
        super(MLP_shallow, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(Din, Nneurons),
            nn.ReLU(),
            nn.Linear(Nneurons, Dout)
        )

    def forward(self, x):

        x = self.layers(x)
        
        return x
## **********


## **********
## Definition of the model as a torch.nn.Model using ModuleList()
class MLP_deep(nn.Module):
    def __init__(self):
        super(MLP_deep, self).__init__()
        
        self.layers =  nn.ModuleList()

        self.layers.append(nn.Linear(Din, Nneurons))
        for i in range(Nhidden):
            self.layers.append(nn.Linear(Nneurons, Nneurons))
        self.layers.append(nn.Linear(Nneurons, Dout))

    def forward(self, x):

        x = torch.flatten(x, 1)
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        x = self.layers[-1](x)
        
        return x
## **********
