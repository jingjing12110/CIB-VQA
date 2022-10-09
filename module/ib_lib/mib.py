# @File  :mib.py
# @Time  :https://github.com/mfederici/Multi-View-Information-Bottleneck
import torch
import torch.nn as nn

from torch.distributions import Normal, Independent
from torch.nn.functional import softplus


class ExponentialScheduler:
    def __init__(self, start_value, end_value, n_iterations,
                 start_iteration=0, base=10):
        self.base = base
        self.start_value = start_value
        self.end_value = end_value
        self.n_iterations = n_iterations
        self.start_iteration = start_iteration
        self.m = (end_value - start_value) / n_iterations
    
    def __call__(self, iteration):
        if iteration > self.start_iteration + self.n_iterations:
            linear_value = self.end_value
        elif iteration <= self.start_iteration:
            linear_value = self.start_value
        else:
            linear_value = (iteration - self.start_iteration
                            ) * self.m + self.start_value
        return self.base ** linear_value


# Encoder architecture
class Encoder(nn.Module):
    def __init__(self, z_dim):
        super(Encoder, self).__init__()
        
        self.z_dim = z_dim
        
        # Vanilla MLP
        self.net = nn.Sequential(
            nn.Linear(28 * 28, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Linear(1024, z_dim * 2),
        )
    
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        params = self.net(x)
        
        mu, sigma = params[:, :self.z_dim], params[:, self.z_dim:]
        sigma = softplus(sigma) + 1e-7  # Make sigma always positive
        # Return a factorized Normal distribution
        return Independent(Normal(loc=mu, scale=sigma), 1)


class Decoder(nn.Module):
    def __init__(self, z_dim, scale=0.39894):
        super(Decoder, self).__init__()
        
        self.z_dim = z_dim
        self.scale = scale
        
        # Vanilla MLP
        self.net = nn.Sequential(
            nn.Linear(z_dim, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 28 * 28)
        )
    
    def forward(self, z):
        x = self.net(z)
        return Independent(Normal(loc=x, scale=self.scale), 1)


# Auxiliary network for mutual information estimation
class MIEstimator(nn.Module):
    def __init__(self, size1, size2, low_dim=1024):
        super(MIEstimator, self).__init__()
        # Vanilla MLP
        self.net = nn.Sequential(
            nn.Linear(size1 + size2, low_dim),
            nn.ReLU(True),
            nn.Linear(low_dim, low_dim),
            nn.ReLU(True),
            nn.Linear(low_dim, 1),
        )
    
    # Gradient for JSD mutual information estimation and EB-based estimation
    def forward(self, x1, x2):
        pos = self.net(torch.cat([x1, x2], 1))  # Positive Samples
        neg = self.net(torch.cat([torch.roll(x1, 1, 0), x2], 1))
        return -softplus(-pos).mean() - softplus(
            neg).mean(), pos.mean() - neg.exp().mean() + 1


######################
# MV InfoMax #
######################
class MVInfoMax(nn.Module):
    def __init__(self, z_dim, **params):
        super(MVInfoMax, self).__init__()
        self.z_dim = z_dim
        
        # Initialization of the mutual information estimation network
        self.mi_estimator = MIEstimator(self.z_dim, self.z_dim)
        
        # Intialization of the encoder(s)
        # In this example encoder_v1 and encoder_v2 completely
        # share their parameters
        self.encoder_v1 = Encoder(z_dim)
        self.encoder_v2 = self.encoder_v1
    
    def forward(self, data):
        # Read the two views v1 and v2 and ignore the label y
        v1, v2, _ = data
        
        # Encode a batch of data
        p_z1_given_v1 = self.encoder_v1(v1)
        p_z2_given_v2 = self.encoder_v2(v2)
        
        # Sample from the posteriors with reparametrization
        z1 = p_z1_given_v1.rsample()
        z2 = p_z2_given_v2.rsample()
        
        # Mutual information estimation
        mi_gradient, mi_estimation = self.mi_estimator(z1, z2)
        mi_gradient = mi_gradient.mean()
        mi_estimation = mi_estimation.mean()
        
        # Computing the loss function
        loss = - mi_gradient
        
        return loss, mi_estimation.item()


###############
# MIB #
###############
class MIB(nn.Module):
    def __init__(self, z_dim, beta_start_value=1e-3, beta_end_value=1,
                 beta_n_iterations=100000, beta_start_iteration=50000):
        # The neural networks architectures and initialization procedure
        # is analogous to Multi-View InfoMax
        super(MIB, self).__init__()
        
        # Initialization of the encoder(s)
        # encoder_v1 and encoder_v2 completely share their parameters
        self.encoder_v1 = Encoder(z_dim)
        self.encoder_v2 = self.encoder_v1
        
        # Initialization of the mutual information estimation network
        self.mi_estimator = MIEstimator(self.z_dim, self.z_dim)
        
        # Definition of the scheduler to update the value of
        # the regularization coefficient beta over time
        self.beta_scheduler = ExponentialScheduler(
            start_value=beta_start_value,
            end_value=beta_end_value,
            n_iterations=beta_n_iterations,
            start_iteration=beta_start_iteration
        )
    
    def forward(self, data, iterations):
        # Read the two views v1 and v2 and ignore the label y
        v1, v2, _ = data
        
        # Encode a batch of data
        p_z1_given_v1 = self.encoder_v1(v1)
        p_z2_given_v2 = self.encoder_v2(v2)
        
        # Sample from the posteriors with reparametrization
        z1 = p_z1_given_v1.rsample()
        z2 = p_z2_given_v2.rsample()
        
        # Mutual information estimation
        mi_gradient, mi_estimation = self.mi_estimator(z1, z2)
        mi_gradient = mi_gradient.mean()
        mi_estimation = mi_estimation.mean()
        
        # Symmetrized Kullback-Leibler divergence
        kl_1_2 = p_z1_given_v1.log_prob(z1) - p_z2_given_v2.log_prob(z1)
        kl_2_1 = p_z2_given_v2.log_prob(z2) - p_z1_given_v1.log_prob(z2)
        skl = (kl_1_2 + kl_2_1).mean() / 2.
        
        # Update the value of beta according to the policy
        beta = self.beta_scheduler(iterations)
        
        # Computing the loss function
        loss = - mi_gradient + beta * skl
        
        return loss, mi_estimation.item(), skl.item()
