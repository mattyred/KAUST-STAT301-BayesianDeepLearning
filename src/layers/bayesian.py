import torch
import torch.nn as nn

class bae_final_linear(nn.Module):
    def __init__(self, in_features, out_features):
        super(bae_final_linear, self).__init__()
        self.mu = nn.Parameter(torch.zeros(out_features, in_features))
        self.rho = nn.Parameter(torch.zeros(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_rho = nn.Parameter(torch.zeros(out_features))

    def forward(self, x, sample=True):
        if sample:
            weight = self.mu + torch.log1p(torch.exp(self.rho)) * torch.randn_like(self.rho)
            bias = self.bias_mu + torch.log1p(torch.exp(self.bias_rho)) * torch.randn_like(self.bias_rho)
        else:
            weight = self.mu
            bias = self.bias_mu

        return nn.functional.linear(x, weight, bias)