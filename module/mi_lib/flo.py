# @File :flo.py
# @Time :2021/12/24
# @Desc :
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class UFunc(nn.Module):
    def __init__(self, x_dim, y_dim, hidden_size=128):
        super(UFunc, self).__init__()
        self.func = nn.Sequential(
            nn.Linear(x_dim + y_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x, y):
        xy = torch.cat([x, y], dim=1)
        
        return self.func(xy)


class BilinearCritic(nn.Module):
    def __init__(self, x_dim, y_dim, hidden_size=512, tau=1., is_u_func=False):
        super(BilinearCritic, self).__init__()
        self.is_u_func = is_u_func
        
        self.encoder_x = nn.Sequential(
            nn.Linear(x_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, x_dim)
        )
        self.encoder_y = nn.Sequential(
            nn.Linear(y_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, y_dim)
        )
        self.log_tau = torch.nn.Parameter(torch.Tensor([np.log(tau)]))
        if self.is_u_func:
            self.u_func = UFunc(x_dim, y_dim)
    
    def forward(self, x, y, tau=None):
        if tau is None:
            tau = torch.exp(self.log_tau)
        tau = torch.sqrt(tau)
        hx = F.normalize(self.encoder_x(x), dim=1)
        hy = F.normalize(self.encoder_y(y), dim=1)
        if self.is_u_func:
            u = self.u_func(hx, hy)
            return hx / tau, hy / tau, u
        else:
            return hx / tau, hy / tau


class FenchelInfoNCE(nn.Module):
    def __init__(self, critic: nn.Module, u_func: nn.Module):
        super(FenchelInfoNCE, self).__init__()
        self.critic = critic
        self.u_func = u_func
    
    def forward(self, x, y, y0):
        """
        :param x: n x p
        :param y: n x d true
        :param y0: n x d fake
        :return:
        """
        output = self.pmi(x, y, y0)
        return output.mean()
    
    def MI(self, x, y, K=10):
        mi = 0
        for k in range(K):
            y0 = y[torch.randperm(y.size()[0])]
            mi += self.forward(x, y, y0)
        return -mi / K
    
    def pmi(self, x, y, y0):
        g = self.critic(x, y)
        g0 = self.critic(x, y0)
        u = self.u_func(x, y)
        output = u + torch.exp(-u + g0 - g) - 1
        return output


class BilinearFDVNCE(nn.Module):
    def __init__(self, x_dim=768, y_dim=768, hidden_size=512, tau=1.):
        super(BilinearFDVNCE, self).__init__()
        self.critic = BilinearCritic(x_dim, y_dim, hidden_size, tau)
        
        self.criterion = torch.nn.CrossEntropyLoss(reduction='none')
    
    def forward(self, x, y, average=True):
        """
        :param average:
        :param x: n x p
        :param y: n x d true
        :return:
        """
        output = self.pmi(x, y)
        output = torch.clamp(output, -5, 15)
        if average:
            return output.mean()
        else:
            return output.sum()
    
    def pmi(self, x, y):
        hx, hy = self.critic(x, y)
        
        similarity_matrix = hx @ hy.t()
        
        pos_mask = torch.eye(hx.size(0), dtype=torch.bool)
        g = similarity_matrix[pos_mask].view(hx.size(0), -1)
        g0 = similarity_matrix[~pos_mask].view(hx.size(0), -1)
        
        logits = g0 - g
        
        slogits = torch.logsumexp(logits, 1).view(-1, 1)
        
        labels = torch.tensor(range(hx.size(0)), dtype=torch.int64).cuda()
        dummy_ce = self.criterion(similarity_matrix, labels) - torch.log(
            torch.Tensor([hx.size(0)]).cuda())
        dummy_ce = dummy_ce.view(-1, 1)
        
        output = dummy_ce.detach() + torch.exp(slogits - slogits.detach()) - 1
        
        return output


class BilinearFenchelInfoNCEOne(nn.Module):
    def __init__(self, x_dim=768, y_dim=768, hidden_size=512, tau=1., K=None):
        super(BilinearFenchelInfoNCEOne, self).__init__()
        self.critic = BilinearCritic(
            x_dim, y_dim, hidden_size, tau, is_u_func=True)
        self.K = K
    
    def forward(self, x, y, y0=None, K=None, average=True):
        """
        :param x: n x p
        :param y: n x d true
        :param y0: n x d fake
        :param K:
        :return:
        """
        if K is None:
            K = self.K
        if y0 is None:
            y0 = y[torch.randperm(y.size()[0])]
        output = self.pmi(x, y, y0, K)
        output = torch.clamp(output, -5, 15)
        
        if average:
            return output.mean()
        else:
            return output.sum()
    
    def pmi(self, x, y, y0=None, K=None):
        # one func mode
        gu = self.critic(x, y)
        if isinstance(gu, tuple):
            hx, hy, u = gu
            similarity_matrix = hx @ hy.t()
            pos_mask = torch.eye(hx.size(0), dtype=torch.bool)
            g = similarity_matrix[pos_mask].view(hx.size(0), -1)
            g0 = similarity_matrix[~pos_mask].view(hx.size(0), -1)
            g0_logsumexp = torch.logsumexp(g0, 1).view(-1, 1)
            output = u + torch.exp(-u + g0_logsumexp - g) / (
                    hx.size(0) - 1) - 1
        else:
            g, u = torch.chunk(self.critic(x, y), 2, dim=1)
            if K is not None:
                for k in range(K - 1):
                    if k == 0:
                        y0 = y0
                        g0, _ = torch.chunk(self.critic(x, y0), 2, dim=1)
                    else:
                        y0 = y[torch.randperm(y.size()[0])]
                        g00, _ = torch.chunk(self.critic(x, y0), 2, dim=1)
                        g0 = torch.cat((g0, g00), 1)
                g0_logsumexp = torch.logsumexp(g0, 1).view(-1, 1)
                output = u + torch.exp(-u + g0_logsumexp - g) / (K - 1) - 1
            else:
                g0, _ = torch.chunk(self.critic(x, y0), 2, dim=1)
                output = u + torch.exp(-u + g0 - g) - 1
        return output
