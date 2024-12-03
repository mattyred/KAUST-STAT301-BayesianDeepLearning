import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence
from torchvision.models import resnet18

class ResNet18(nn.Module):
    def __init__(self, base_model, dropout_rate=0.5, output_dim=2):
        super(ResNet18, self).__init__()
        self.dropout_rate = dropout_rate

        self.modified_layers = self.add_dropout_to_bn2(base_model) # add dropout after second BatchNorm2d in each BasicBlock

        in_features = base_model.fc.in_features
        self.modified_layers.fc = nn.Linear(in_features, output_dim)

    def add_dropout_to_bn2(self, model):
        for name, module in model.named_children():
            if name.startswith('layer'):
                for block_name, block_module in module.named_children():
                    if hasattr(block_module, 'bn2') and isinstance(block_module.bn2, nn.BatchNorm2d):
                        block_module.bn2 = nn.Sequential(
                            block_module.bn2,
                            nn.Dropout(self.dropout_rate)
                        )

            elif len(list(module.children())) > 0:
                self.add_dropout_to_bn2(module)
        return model

    def forward(self, x):
        x = self.modified_layers(x) 
        return x

class BaeFinalLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(BaeFinalLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.mu = nn.Parameter(torch.zeros(out_features, in_features))
        self.log_sigma = nn.Parameter(torch.ones(out_features, in_features) * -2)  # Std ~ 0.135
        self.prior = Normal(0, 1)

    def forward(self, x):
        sigma = torch.exp(self.log_sigma)
        eps = torch.randn_like(self.mu)
        weight = self.mu + sigma * eps
        
        return x @ weight.t()

    def kl_divergence(self):
        posterior = Normal(self.mu, torch.exp(self.log_sigma))
        return kl_divergence(posterior, self.prior).sum()
        
class BaeResNet18(nn.Module):
    def __init__(self, num_classes=2):
        super(BaeResNet18, self).__init__()
        
        self.resnet = resnet18(pretrained=True)
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()
        self.bayesian_fc = BaeFinalLinear(in_features, num_classes)

    def forward(self, x):
        x = self.resnet(x)
        return self.bayesian_fc(x)

    def kl_divergence(self):
        return self.bayesian_fc.kl_divergence()