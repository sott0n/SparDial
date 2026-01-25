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


class MMNet(torch.nn.Module):
    """Matrix multiplication network"""

    def forward(self, x, y):
        return torch.mm(x, y)


class MVNet(torch.nn.Module):
    """Matrix-vector multiplication network"""

    def forward(self, x, v):
        return torch.mv(x, v)


class SDDMMNet(torch.nn.Module):
    """Sampled Dense-Dense Matrix Multiplication"""

    def forward(self, x, y, z):
        return torch.mul(x, torch.mm(y, z))


class Normalization(torch.nn.Module):
    """Graph normalization: D^-1 @ A @ D^-1"""

    def forward(self, A):
        sum_vector = torch.sum(A, dim=1)
        reciprocal_vector = 1 / sum_vector
        reciprocal_vector[reciprocal_vector == float("inf")] = 0
        scaling_diagonal = torch.diag(reciprocal_vector).to_sparse()
        return scaling_diagonal @ A @ scaling_diagonal


class SimpleLinear(torch.nn.Module):
    """Simple linear layer"""

    def __init__(self, in_features=4, out_features=4):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x)
