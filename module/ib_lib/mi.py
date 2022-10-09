# @File :mi.py
# @Time :2021/7/2
# @Desc :
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import softplus
from torch.distributions import Normal, Independent

from module.mi_lib.lower_bound import CPC


class MINE(nn.Module):
    def __init__(self, x_dim, y_dim, hidden_size):
        super(MINE, self).__init__()
        self.T_func = nn.Sequential(
            nn.Linear(x_dim + y_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, x_samples, y_samples):
        # samples have shape [sample_size, dim]
        # shuffle and concatenate
        sample_size = y_samples.shape[0]
        random_index = torch.randperm(sample_size).long()
        
        y_shuffle = y_samples[random_index]
        
        T0 = self.T_func(torch.cat([x_samples, y_samples], dim=-1))
        T1 = self.T_func(torch.cat([x_samples, y_shuffle], dim=-1))
        
        lower_bound = T0.mean() - torch.log(T1.exp().mean())
        
        # compute the negative loss (maximise loss == minimise -loss)
        return lower_bound


class NWJ(nn.Module):
    def __init__(self, x_dim, y_dim, hidden_size=768):
        super(NWJ, self).__init__()
        self.F_func = nn.Sequential(
            nn.Linear(x_dim + y_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, x_samples, y_samples):
        # shuffle and concatenate
        sample_size = y_samples.shape[0]
        
        x_tile = x_samples.unsqueeze(0).repeat((sample_size, 1, 1))
        y_tile = y_samples.unsqueeze(1).repeat((1, sample_size, 1))
        
        T0 = self.F_func(torch.cat([x_samples, y_samples], dim=-1))
        # shape [sample_size, sample_size, 1]
        T1 = self.F_func(torch.cat([x_tile, y_tile], dim=-1)) - 1.
        
        lower_bound = T0.mean() - (
                T1.logsumexp(dim=1) - np.log(sample_size)).exp().mean()
        return lower_bound


class VarUB(nn.Module):
    def __init__(self, x_dim, y_dim, hidden_size):
        """variational upper bound
        """
        super(VarUB, self).__init__()
        self.p_mu = nn.Sequential(
            nn.Linear(x_dim, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, y_dim)
        )
        
        self.p_logvar = nn.Sequential(
            nn.Linear(x_dim, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, y_dim),
            nn.Tanh()
        )
    
    def forward(self, x_samples, y_samples):  # [sample_size, 1]
        mu, logvar = self.p_mu(x_samples), self.p_logvar(x_samples)
        return 1. / 2. * (mu ** 2 + logvar.exp() - 1. - logvar).mean()


class L1OutUB(nn.Module):
    def __init__(self, x_dim=768, y_dim=768, hidden_size=768):
        """naive upper bound
        """
        super(L1OutUB, self).__init__()
        self.p_mu = nn.Sequential(
            nn.Linear(x_dim, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, y_dim)
        )
        
        self.p_logvar = nn.Sequential(
            nn.Linear(x_dim, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, y_dim),
            nn.Tanh()
        )
    
    @staticmethod
    def log_sum_exp(value, dim=None, keepdim=False):
        """Numerically stable implementation of the operation
        value.exp().sum(dim, keepdim).log()
        """
        import math
        if dim is not None:
            m, _ = torch.max(value, dim=dim, keepdim=True)
            value0 = value - m
            if keepdim is False:
                m = m.squeeze(dim)
            return m + torch.log(torch.sum(
                torch.exp(value0), dim=dim, keepdim=keepdim))
        else:
            m = torch.max(value)
            sum_exp = torch.sum(torch.exp(value - m))
            if isinstance(sum_exp, Number):
                return m + math.log(sum_exp)
            else:
                return m + torch.log(sum_exp)
    
    def forward(self, x_samples, y_samples):
        batch_size = y_samples.shape[0]
        
        mu, logvar = self.p_mu(x_samples), self.p_logvar(x_samples)
        
        # [sample_size]
        positive = (- (mu - y_samples) ** 2 / 2. / logvar.exp() - logvar / 2.
                    ).sum(dim=-1)
        
        # [sample_size, 1, dim]
        mu_1 = mu.unsqueeze(1)
        logvar_1 = logvar.unsqueeze(1)
        # [1, sample_size, dim]
        y_samples_1 = y_samples.unsqueeze(0)
        # [sample_size, sample_size]
        all_probs = (- (y_samples_1 - mu_1
                        ) ** 2 / 2. / logvar_1.exp() - logvar_1 / 2.).sum(dim=-1)
        
        diag_mask = torch.ones([batch_size]).diag().unsqueeze(-1).cuda() * (-20.)
        # [sample_size]
        
        negative = self.log_sum_exp(all_probs + diag_mask, dim=0) - np.log(
            batch_size - 1.)
        
        return (positive - negative).mean()
    
    def update(self, x_samples):
        batch_size = x_samples.shape[0]
        mu, logvar = self.p_mu(x_samples), self.p_logvar(x_samples)

        random_index = torch.randperm(batch_size).long()
        y_samples = x_samples[random_index]
        
        # [sample_size]
        positive = (- (mu - y_samples) ** 2 / 2. / logvar.exp() - logvar / 2.
                    ).sum(dim=-1)
        # [sample_size, 1, dim]
        mu_1 = mu.unsqueeze(1)
        logvar_1 = logvar.unsqueeze(1)
        # [1, sample_size, dim]
        y_samples_1 = y_samples.unsqueeze(0)
        # [sample_size, sample_size]
        all_probs = (- (y_samples_1 - mu_1
                        ) ** 2 / 2. / logvar_1.exp() - logvar_1 / 2.).sum(dim=-1)

        diag_mask = torch.ones([batch_size]).diag().unsqueeze(-1).cuda() * (-20.)
        # [sample_size]

        negative = self.log_sum_exp(all_probs + diag_mask, dim=0) - np.log(
            batch_size - 1.)

        return (positive - negative).mean()


class InfoNCE(nn.Module):
    def __init__(self, x_dim=768, y_dim=768, hidden_size=300):
        super(InfoNCE, self).__init__()
        self.F_func = nn.Sequential(
            nn.Linear(x_dim + y_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Softplus()
        )
    
    def forward(self, x_samples, y_samples):
        # samples have shape [sample_size, dim]
        # shuffle and concatenate
        sample_size = y_samples.shape[0]
        random_index = torch.randperm(sample_size).long()
        
        x_tile = x_samples.unsqueeze(0).repeat((sample_size, 1, 1))
        y_tile = y_samples[random_index].unsqueeze(1).repeat((1, sample_size, 1))
        
        T0 = self.F_func(torch.cat([x_samples, y_samples], dim=-1))
        T1 = self.F_func(
            torch.cat([x_tile, y_tile], dim=-1))  # [s_size, s_size, 1]
        
        lower_bound = T0.mean() - (
                T1.logsumexp(dim=1).mean() - np.log(sample_size))
        
        # compute the negative loss (maximise loss == minimise -loss)
        return lower_bound


# ******************************************************************************
# Modifying for CIB
# ******************************************************************************


# Auxiliary network for mutual information estimation
class MIEstimator(nn.Module):
    def __init__(self, size1, size2):
        super(MIEstimator, self).__init__()
        # Vanilla MLP
        self.net = nn.Sequential(
            nn.Linear(size1 + size2, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1),
        )
    
    # Gradient for JSD mutual information estimation and EB-based estimation
    def forward(self, x1, x2):
        pos = self.net(torch.cat([x1, x2], 1))  # Positive Samples
        neg = self.net(torch.cat([torch.roll(x1, 1, 0), x2], 1))
        return -softplus(-pos).mean() - softplus(
            neg).mean(), pos.mean() - neg.exp().mean() + 1


class MVMIEstimator(nn.Module):
    def __init__(self, x1_dim=768, x2_dim=768, hidden_size=384, lb_name=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.lb_name = lb_name
        if lb_name == "JSD":
            self.mi_estimator = MIEstimator(x1_dim // 2, x2_dim // 2)
        elif lb_name == "CPC":
            self.mi_estimator = CPC(x1_dim // 2, x2_dim // 2)
        elif lb_name == "InfoNCE":
            self.mi_estimator = InfoNCE(
                x1_dim // 2, x2_dim // 2, hidden_size)
        elif lb_name == "NWJ":
            self.mi_estimator = NWJ(
                x1_dim // 2, x2_dim // 2, hidden_size)
        elif lb_name == "MINE":
            self.mi_estimator = MINE(
                x1_dim // 2, x2_dim // 2, hidden_size)
        else:
            raise ModuleNotFoundError
    
    def forward(self, p_z1_given_x1, p_z2_given_x2):
        mu = p_z1_given_x1[:, :self.hidden_size]
        sigma = p_z1_given_x1[:, self.hidden_size:]
        # Make sigma always positive
        sigma = softplus(sigma) + 1e-7
        # a factorized Normal distribution
        p_z1_given_x1 = Independent(Normal(loc=mu, scale=sigma), 1)
        
        mu = p_z2_given_x2[:, :self.hidden_size]
        sigma = p_z2_given_x2[:, self.hidden_size:]
        sigma = softplus(sigma) + 1e-7
        # a factorized Normal distribution
        p_z2_given_x2 = Independent(Normal(loc=mu, scale=sigma), 1)
        
        # Sample from the posteriors with reparametrization
        z1 = p_z1_given_x1.rsample()
        z2 = p_z2_given_x2.rsample()

        # Symmetrized Kullback-Leibler divergence
        kl_1_2 = p_z1_given_x1.log_prob(z1) - p_z2_given_x2.log_prob(z1)
        kl_2_1 = p_z2_given_x2.log_prob(z2) - p_z1_given_x1.log_prob(z2)
        d_skl_2 = (kl_1_2 + kl_2_1).mean() / 2.
        
        # Mutual information estimation
        if self.lb_name == "JSD":
            mi_gradient, mi_estimation = self.mi_estimator(z1, z2)
            return -mi_gradient, d_skl_2
        else:
            mi_estimation = self.mi_estimator(z1, z2)
            return mi_estimation, d_skl_2

    def forward_skl(self, p_z1_given_x1, p_z2_given_x2):
        mu = p_z1_given_x1[:, :self.hidden_size]
        sigma = p_z1_given_x1[:, self.hidden_size:]
        # Make sigma always positive
        sigma = softplus(sigma) + 1e-7
        # a factorized Normal distribution
        p_z1_given_x1 = Independent(Normal(loc=mu, scale=sigma), 1)
    
        mu = p_z2_given_x2[:, :self.hidden_size]
        sigma = p_z2_given_x2[:, self.hidden_size:]
        sigma = softplus(sigma) + 1e-7
        # a factorized Normal distribution
        p_z2_given_x2 = Independent(Normal(loc=mu, scale=sigma), 1)
    
        # Sample from the posteriors with reparametrization
        z1 = p_z1_given_x1.rsample()
        z2 = p_z2_given_x2.rsample()
    
        # Mutual information estimation
        if self.lb_name == "JSD":
            mi_gradient, mi_estimation = self.mi_estimator(z1, z2)
        else:
            mi_estimation = self.mi_estimator(z1, z2)
        
        # Symmetrized Kullback-Leibler divergence
        kl_1_2 = F.kl_div(
            F.log_softmax(z1, dim=-1), F.softmax(z2, dim=-1),
            reduction='batchmean')
        kl_2_1 = F.kl_div(
            F.log_softmax(z2, dim=-1), F.softmax(z1, dim=-1),
            reduction='batchmean')
        d_skl_2 = (kl_1_2 + kl_2_1).mean() / 2.
        
        return mi_estimation, d_skl_2

    def compute_mi(self, p_z1_given_x1, p_z2_given_x2):
        mu = p_z1_given_x1[:, :self.hidden_size]
        sigma = p_z1_given_x1[:, self.hidden_size:]
        # Make sigma always positive
        sigma = softplus(sigma) + 1e-7
        # a factorized Normal distribution
        p_z1_given_x1 = Independent(Normal(loc=mu, scale=sigma), 1)
    
        mu = p_z2_given_x2[:, :self.hidden_size]
        sigma = p_z2_given_x2[:, self.hidden_size:]
        sigma = softplus(sigma) + 1e-7
        # a factorized Normal distribution
        p_z2_given_x2 = Independent(Normal(loc=mu, scale=sigma), 1)
    
        # Sample from the posteriors with re-parametrization
        z1 = p_z1_given_x1.rsample()
        z2 = p_z2_given_x2.rsample()
    
        # Mutual information estimation
        mi_gradient, mi_estimation = self.mi_estimator(z1, z2)
        mi_gradient = mi_gradient.mean()
        return -mi_gradient


