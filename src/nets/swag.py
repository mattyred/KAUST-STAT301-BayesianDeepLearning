import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from .resnet import myresnet18

class SWAG_ResNet18(nn.Module):
    def __init__(self, **kwargs):
        """
            Implementation of diag-SWA model based on pseudocode from SWAG paper
            (https://arxiv.org/pdf/1902.02476)
        """
        super(SWAG_ResNet18, self).__init__()
        
        self.base_model = myresnet18(**kwargs)
        
        self.theta_mean = {}  
        self.theta2_mean = {}  
        self.n = 0  
        
        # Initialize mean and second moment for each base_model parameter
        for name, param in self.base_model.state_dict().items():
            self.theta_mean[name] = torch.zeros_like(param)
            self.theta2_mean[name] = torch.zeros_like(param)

    def forward(self, x):
        return self.base_model(x)
    
    def update_moments(self):
        state_dict = self.base_model.state_dict()  # Get current model parameters
        
        for name, param in state_dict.items():
            device = param.device
            
            self.theta_mean[name] = self.theta_mean[name].to(device) 
            self.theta2_mean[name] = self.theta2_mean[name].to(device) 
            
            # Update running mean
            prev_mean = self.theta_mean[name]
            self.theta_mean[name] = (self.n * prev_mean + param) / (self.n + 1)
            
            # Update second moment E[x^2]
            self.theta2_mean[name] = (self.n * self.theta2_mean[name] + param ** 2) / (self.n + 1)

        self.n += 1 

    def sample(self):
        """
        Sample weights from the SWAG Gaussian distribution for each parameter.
        """
        sampled_state_dict = copy.deepcopy(self.base_model.state_dict())
        
        for name, param in sampled_state_dict.items():
            device = param.device
            self.theta_mean[name] = self.theta_mean[name].to(device)
            self.theta2_mean[name] = self.theta2_mean[name].to(device)
            
            # Compute variance (Var[x] = E[x^2] - (E[x])^2)
            var = self.theta2_mean[name] - self.theta_mean[name] ** 2
            var = torch.clamp(var, min=1e-30)
            
            sampled_param = torch.normal(mean=self.theta_mean[name], std=torch.sqrt(var))
            sampled_state_dict[name].copy_(sampled_param)

        self.base_model.load_state_dict(sampled_state_dict)

class SWAG(nn.Module):
    def __init__(self, model, **kwargs):
        """
            Implementation of diag-SWA model based on pseudocode from SWAG paper
            (https://arxiv.org/pdf/1902.02476)
        """
        super(SWAG, self).__init__()
        
        self.base_model = model
        
        self.theta_mean = {}  
        self.theta2_mean = {}  
        self.n = 0  
        
        # Initialize mean and second moment for each base_model parameter
        for name, param in self.base_model.state_dict().items():
            self.theta_mean[name] = torch.zeros_like(param)
            self.theta2_mean[name] = torch.zeros_like(param)

    def forward(self, x):
        return self.base_model(x)
    
    def update_moments(self):
        state_dict = self.base_model.state_dict()  # Get current model parameters
        
        for name, param in state_dict.items():
            device = param.device
            
            self.theta_mean[name] = self.theta_mean[name].to(device) 
            self.theta2_mean[name] = self.theta2_mean[name].to(device) 
            
            # Update running mean
            prev_mean = self.theta_mean[name]
            self.theta_mean[name] = (self.n * prev_mean + param) / (self.n + 1)
            
            # Update second moment E[x^2]
            self.theta2_mean[name] = (self.n * self.theta2_mean[name] + param ** 2) / (self.n + 1)

        self.n += 1 

    def sample(self):
        """
        Sample weights from the SWAG Gaussian distribution for each parameter.
        """
        sampled_state_dict = copy.deepcopy(self.base_model.state_dict())
        
        for name, param in sampled_state_dict.items():
            device = param.device
            self.theta_mean[name] = self.theta_mean[name].to(device)
            self.theta2_mean[name] = self.theta2_mean[name].to(device)
            
            # Compute variance (Var[x] = E[x^2] - (E[x])^2)
            var = self.theta2_mean[name] - self.theta_mean[name] ** 2
            var = torch.clamp(var, min=1e-30)
            
            sampled_param = torch.normal(mean=self.theta_mean[name], std=torch.sqrt(var))
            sampled_state_dict[name].copy_(sampled_param)

        self.base_model.load_state_dict(sampled_state_dict)