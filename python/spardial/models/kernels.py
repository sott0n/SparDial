"""Sample PyTorch Models for Testing"""

import torch


class AddNet(torch.nn.Module):
    """Simple addition network"""
    def forward(self, x, y):
        return torch.add(x, y)


class MulNet(torch.nn.Module):
    """Simple multiplication network"""
    def forward(self, x, y):
        return torch.mul(x, y)


class SimpleLinear(torch.nn.Module):
    """Simple linear layer"""
    def __init__(self, in_features=4, out_features=4):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x)
