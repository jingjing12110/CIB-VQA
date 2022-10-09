# @File :upper_bound.py
# @Time :2021/12/18
# @Desc :
import torch
import torch.nn as nn


class CLUB(nn.Module):
    """Compute the Contrastive Log-ratio Upper Bound (CLUB) given input pair
        Args:
            hidden_size(int): embedding size
    """
    
    def __init__(self, x_dim=768, y_dim=768, hidden_size=768):
        super(CLUB, self).__init__()
        # p_mu outputs mean of q(Y|X)
        # self.p_mu = nn.Sequential(
        #     nn.Linear(x_dim, hidden_size // 2),
        #     nn.ReLU(),
        #     nn.Linear(hidden_size // 2, y_dim)
        # )
        # # p_logvar outputs log of variance of q(Y|X)
        # self.p_logvar = nn.Sequential(
        #     nn.Linear(x_dim, hidden_size // 2),
        #     nn.ReLU(),
        #     nn.Linear(hidden_size // 2, y_dim),
        #     nn.Tanh()
        # )
    
    def forward(self, x, y):
        """
            CLUB with random shuffle, the Q function in original paper:
                CLUB = E_p(x,y)[log q(y|x)]-E_p(x)p(y)[log q(y|x)]
            Args:
                x (Tensor): x in above equation
                y (Tensor): y in above equation
        """
        # mu, logvar = self.p_mu(x), self.p_logvar(x)  # [bs, dim]
        
        bs = y.size(0)
        random_index = torch.randperm(bs).long()
        
        # # pred v using l
        # pred_tile = mu.unsqueeze(1).repeat(1, bs, 1)  # (bs, bs, emb_size)
        # true_tile = y.unsqueeze(0).repeat(bs, 1, 1)  # (bs, bs, emb_size)
        #
        # positive = - (mu - y) ** 2 / 2. / logvar.exp()
        # # log of conditional probability of negative sample pairs
        # negative = - ((true_tile - pred_tile) ** 2).mean(
        #     dim=1) / 2. / logvar.exp()
        #
        # # lld = torch.mean(torch.sum(positive, -1))
        # upper_bound = (positive.sum(dim=-1) - negative.sum(dim=-1)).mean()
        # return upper_bound  # lld,
        
        # log of conditional probability of positive sample pairs
        # positive = - (mu - y) ** 2 / 2. / logvar.exp()
        
        # log of conditional probability of negative sample pairs
        # negative = - ((mu - y[random_index]) ** 2) / 2. / logvar.exp()
        positive = torch.zeros_like(y)
        negative = - (y - y[random_index]) ** 2 / 2.
        
        upper_bound = (positive.sum(dim=-1) - negative.sum(dim=-1)).mean()
        return upper_bound
