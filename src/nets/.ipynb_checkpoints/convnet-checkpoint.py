import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

class SWAG_CNN(nn.Module):
    def __init__(self, K=10):
        super(SWAG_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(64*12*12, 128)
        self.fc2 = nn.Linear(128, 10)

        self.K = K  
        self.mean = {}
        self.variance = {}
        self.num_snapshots = 0
        
        # Initialize the buffers for mean and variance as zero tensors for each parameter
        for name, param in self.state_dict().items():
            self.mean[name] = torch.zeros_like(param)  # Initialize mean with zeros
            self.variance[name] = torch.zeros_like(param)  # Initialize variance with zeros

    def collect_model(self):
        """
        Collect model's weights into SWAG buffers (mean and variance).
        """
        # Update mean and variance based on current state_dict
        state_dict = self.state_dict()
        for name, param in state_dict.items():
            device = param.device
            self.mean[name] = self.mean[name].to(device)
            self.variance[name] = self.variance[name].to(device)
            
            self.mean[name] = (self.num_snapshots * self.mean[name] + param) / (self.num_snapshots + 1)
            if self.num_snapshots > 0:
                self.variance[name] = (self.num_snapshots * self.variance[name] +
                                       (param - self.mean[name]) ** 2) / (self.num_snapshots + 1)
        self.num_snapshots += 1
    
    def sample(self, scale=1.0):
        """
        Sample a new set of weights from the Gaussian approximation of the posterior.
        """
        sampled_state_dict = copy.deepcopy(self.state_dict())
        for name, param in sampled_state_dict.items():
            variance = self.variance[name]
            sampled_param = self.mean[name] + scale * torch.randn_like(param) * torch.sqrt(variance)
            sampled_state_dict[name].copy_(sampled_param)
        self.load_state_dict(sampled_state_dict)  # Load the sampled weights

    
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)  # 2x2 max pooling
        x = torch.flatten(x, 1)  # Flatten to feed into fully connected layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)  # Log softmax for classification