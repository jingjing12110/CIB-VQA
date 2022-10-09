# @File :hgib.py
# @Github :https://github.com/wufan2021/Heterogeneous-Graph-Information-Bottleneck
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import softplus

from module.graph_lib.gcn import GCN


class MIEstimator(nn.Module):
    def __init__(self, size1, size2):
        """Mutual Information Estimation"""
        super(MIEstimator, self).__init__()
        # 多层感知机
        self.net = nn.Sequential(
            nn.Linear(size1 + size2, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 1),
        )
    
    # 互信息的梯度与散度
    # Gradient for JSD mutual information estimation and EB-based estimation
    def forward(self, x1, x2):
        pos = self.net(torch.cat([x1, x2], 1))  # Positive Samples
        neg = self.net(torch.cat([torch.roll(x1, 1, 0), x2], 1))
        return -softplus(-pos).mean() - softplus(
            neg).mean(), pos.mean() - neg.exp().mean() + 1


class Encoder(nn.Module):
    def __init__(self, in_feats, n_hidden1=512, n_hidden2=128, activation=F.relu):
        super(Encoder, self).__init__()
        self.base_gcn = GCN(in_feats, n_hidden1, activation)
        self.mean_gcn = GCN(n_hidden1, n_hidden2,
                            act=lambda x: x)
    
    def forward(self, g, x):
        h = self.base_gcn(g, x)
        mean = self.mean_gcn(g, h)
        return mean
    
    
class HGIB(nn.Module):
    def __init__(self, g1, g2, fts, n_hidden2=128, beta=1e-3):
        super(HGIB, self).__init__()
        self.g1 = g1
        self.g2 = g2
        self.beta = beta
        self.encoder_v1 = Encoder(fts.shape[1])
        # self.encoder_v2 = Encoder()
        self.encoder_v2 = self.encoder_v1
        self.mi_estimator = MIEstimator(n_hidden2, n_hidden2)
        self.kl_estimator_1 = MIEstimator(n_hidden2, n_hidden2)
        self.kl_estimator_2 = MIEstimator(n_hidden2, n_hidden2)

        self.loss = None
    
    def forward(self, x1, x2):
        # view1的embedding，其pooling后的结果为v1，view2同理
        z1 = self.encoder_v1(self.g1, x1)
        v1 = torch.mean(z1, dim=0)
        v1 = v1.expand_as(z1)
        
        z2 = self.encoder_v2(self.g2, x2)
        v2 = torch.mean(z2, dim=0)
        v2 = v2.expand_as(z2)
        
        mi_gradient, mi_estimation = self.mi_estimator(z1, z2)
        mi_gradient = mi_gradient.mean()
        mi_estimation = mi_estimation.mean()
        
        skl_v1_z2, _ = self.kl_estimator_1(v1, z2)
        skl_v2_z1, _ = self.kl_estimator_2(v2, z1)
        skl = skl_v1_z2 + skl_v2_z1
        skl = skl.mean()
        
        self.loss = -mi_gradient + self.beta * skl
        
        return mi_estimation
    
    def compute_loss(self):
        return self.loss
    
    def embed(self, fts, z1w, z2w):
        z1 = self.encoder_v1(self.g1, fts)
        z2 = self.encoder_v2(self.g2, fts)
        return z1 * z1w + z2 * z2w, z1, z2
