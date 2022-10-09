# @File :lower_bound.py
# @Time :2021/12/18
# @Desc :
import numpy as np
import torch
import torch.nn as nn


class CPC(nn.Module):
    """Contrastive Predictive Coding: score computation.
        Args:
            x_size (int): embedding size of input modality representation x
            y_size (int): embedding size of input modality representation y
    """
    def __init__(self, x_size, y_size, n_layers=1, activation='Tanh'):
        super().__init__()
        self.x_size = x_size
        self.y_size = y_size
        self.layers = n_layers
        self.activation = getattr(nn, activation)
        if n_layers == 1:
            self.net = nn.Linear(
                in_features=y_size,
                out_features=x_size
            )
        else:
            net = []
            for i in range(n_layers):
                if i == 0:
                    net.append(nn.Linear(self.y_size, self.x_size))
                    net.append(self.activation())
                else:
                    net.append(nn.Linear(self.x_size, self.x_size))
            self.net = nn.Sequential(*net)
    
    def forward(self, x, y, average=True):
        """Calculate the score
        """
        x_pred = self.net(y)  # [bs, dim]
        
        # normalize to unit sphere
        x_pred = x_pred / x_pred.norm(dim=1, keepdim=True)
        x = x / x.norm(dim=1, keepdim=True)
        
        pos = torch.sum(x * x_pred, dim=-1)  # bs
        neg = torch.logsumexp(torch.matmul(x, x_pred.t()), dim=-1)  # bs
        nce = -(pos - neg)
        if average:
            return nce.mean()
        else:
            return nce.sum()
