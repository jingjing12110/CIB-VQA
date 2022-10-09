# @File  :nonlinear_ib.py
# @Github  :https://github.com/burklight/nonlinear-IB-PyTorch
import torch


class DeterministicEncoder(torch.nn.Module):
    """Probabilistic encoder of the network.
    - We use the one in Kolchinsky et al. 2019 "Nonlinear Information Bottleneck"
    - Parameters:
        · K (int) : dimensionality of the bottleneck variable
        · n_x (int) : dimensionality of the input variable
    """
    def __init__(self, K, n_x):
        super(DeterministicEncoder, self).__init__()
        self.K = K
        self.n_x = n_x
        
        layers = [torch.nn.Linear(n_x, 128),
                  torch.nn.ReLU(),
                  torch.nn.Linear(128, 128),
                  torch.nn.ReLU(),
                  torch.nn.Linear(128, self.K)]
        self.f_theta = torch.nn.Sequential(*layers)
    
    def forward(self, x):
        x = x.view(-1, self.n_x)
        mean_t = self.f_theta(x)
        return mean_t


class DeterministicDecoder(torch.nn.Module):
    """Deterministic decoder of the network.
    - We use the one in Kolchinsky et al. 2019 "Nonlinear Information Bottleneck"
    - Parameters:
        · K (int) : dimensionality of the bottleneck variable
        · n_y (int) : dimensionality of the output variable (number of classes)
    """
    def __init__(self, K, n_y):
        super(DeterministicDecoder, self).__init__()
        self.K = K
        
        layers = [torch.nn.Linear(self.K, 128),
                  torch.nn.ReLU(),
                  torch.nn.Linear(128, n_y)]
        self.g_theta = torch.nn.Sequential(*layers)
    
    def forward(self, t):
        logits_y = self.g_theta(t).squeeze()
        return logits_y


class NonlinearIB(torch.nn.Module):
    def __init__(self, K, n_x, n_y, logvar_t=-1.0, train_logvar_t=False):
        """Nonlinear Information Bottleneck network.
        Kolchinsky et al. 2019 "Nonlinear Information Bottleneck"
        @param K: int, dimensionality of the bottleneck variable
        @param n_x: int, dimensionality of the input variable
        @param n_y: int, dimensionality of the output variable (number of classes)
        @param logvar_t:
        @param train_logvar_t: bool, if true, logvar_t is trained
        """
        super(NonlinearIB, self).__init__()
        
        self.encoder = DeterministicEncoder(K, n_x)
        self.decoder = DeterministicDecoder(K, n_y)
        if train_logvar_t:
            self.logvar_t = torch.nn.Parameter(torch.Tensor([logvar_t]))
        else:
            self.logvar_t = torch.Tensor([logvar_t])
    
    def encode(self, x, random=True):
        mean_t = self.encoder(x)
        if random:
            t = mean_t + torch.exp(0.5 * self.logvar_t) * torch.randn_like(mean_t)
        else:
            t = mean_t
        return t
    
    def apply_noise(self, mean_t):
        return mean_t + torch.exp(0.5 * self.logvar_t) * torch.randn_like(mean_t)
    
    def decode(self, t):
        logits_y = self.decoder(t)
        return logits_y
    
    def forward(self, x):
        t = self.encode(x)
        return self.decode(t)
