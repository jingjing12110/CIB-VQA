# @File  :graph_normalization.py
# @Desc  :https://github.com/Kaixiong-Zhou/DGN.git
import torch
import torch.nn as nn
import torch.nn.functional as F


class GroupNorm(nn.Module):
    """modify BatchNorm for 3D inputs"""
    def __init__(self, dim_hidden,  skip_connect=True,
                 num_groups=6, skip_weight=0.005):
        super(GroupNorm, self).__init__()
        self.dim_hidden = dim_hidden
        self.skip_connect = skip_connect
        self.num_groups = num_groups
        self.skip_weight = skip_weight

        self.group_func = nn.Linear(dim_hidden, self.num_groups, bias=True)
        self.bn = nn.BatchNorm1d(dim_hidden * self.num_groups, momentum=0.3)
        
    def forward(self, x):
        """
        :param x: [bs, n_node, dim_hidden]
        :return:
        """
        bs = x.shape[0]
        if self.num_groups == 1:
            x_temp = self.bn(x)
        else:
            score_cluster = F.softmax(self.group_func(x), dim=-1)
            x_temp = torch.cat(
                [score_cluster[:, :, group].unsqueeze(dim=2) * x for group in
                 range(self.num_groups)], dim=-1)
            x_temp = self.bn(x_temp.permute(0, 2, 1)).permute(0, 2, 1).view(
                bs, -1, self.num_groups, self.dim_hidden).sum(dim=2)
        x = x + x_temp * self.skip_weight
        return x


class BatchNorm(nn.Module):
    """Towards Deeper Graph Neural Networks with Differentiable Group 
    Normalization. Zhou, Kaixiong, et al."""
    def __init__(self, dim_hidden, type_norm='group', skip_connect=True,
                 num_groups=6, skip_weight=0.005):
        """
        @param dim_hidden:
        @param type_norm: ['None', 'batch', 'pair', 'group']
        @param skip_connect:
        @param num_groups:
        @param skip_weight:
        """
        super(BatchNorm, self).__init__()
        self.type_norm = type_norm
        self.skip_connect = skip_connect
        self.num_groups = num_groups
        self.skip_weight = skip_weight
        self.dim_hidden = dim_hidden
        if self.type_norm == 'batch':
            self.bn = nn.BatchNorm1d(dim_hidden, momentum=0.3)
        elif self.type_norm == 'group':
            self.bn = nn.BatchNorm1d(dim_hidden * self.num_groups, momentum=0.3)
            self.group_func = nn.Linear(dim_hidden, self.num_groups, bias=True)
        else:
            pass
    
    def forward(self, x):
        if self.type_norm == 'None':
            return x
        elif self.type_norm == 'batch':
            return self.bn(x)
        elif self.type_norm == 'pair':
            col_mean = x.mean(dim=0)
            x = x - col_mean
            rownorm_mean = (1e-6 + x.pow(2).sum(dim=1).mean()).sqrt()
            x = x / rownorm_mean
            return x
        elif self.type_norm == 'group':
            if self.num_groups == 1:
                x_temp = self.bn(x)
            else:
                score_cluster = F.softmax(self.group_func(x), dim=1)
                x_temp = torch.cat(
                    [score_cluster[:, group].unsqueeze(dim=1) * x for group in
                     range(self.num_groups)], dim=1)
                x_temp = self.bn(x_temp).view(
                    -1, self.num_groups, self.dim_hidden).sum(dim=1)
            x = x + x_temp * self.skip_weight
            return x
        else:
            raise Exception(f'the normalization has not been implemented')


class PairNorm(nn.Module):
    """PairNorm: https://github.com/LingxiaoShawn/PairNorm."""
    def __init__(self, mode='PN', scale=1):
        """ 'SCS'-mode is not in the paper but we found it works well in practice,
        especially for GCN and GAT.
        @param mode:
            'None' : No normalization
            'PN'   : Original version
            'PN-SI'  : Scale-Individually version
            'PN-SCS' : Scale-and-Center-Simultaneously version
        @param scale:
        """
        assert mode in ['None', 'PN', 'PN-SI', 'PN-SCS']
        super(PairNorm, self).__init__()
        self.mode = mode
        # Scale can be set based on original data,
        # and also the current feature lengths.
        # We leave the experiments to future.
        # A good pool we used for choosing scale: [0.1, 1, 10, 50, 100]
        self.scale = scale
    
    def forward(self, x):
        if self.mode == 'None':
            return x
        
        col_mean = x.mean(dim=0)
        if self.mode == 'PN':
            x = x - col_mean
            rownorm_mean = (1e-6 + x.pow(2).sum(dim=1).mean()).sqrt()
            x = self.scale * x / rownorm_mean
        
        if self.mode == 'PN-SI':
            x = x - col_mean
            rownorm_individual = (1e-6 + x.pow(2).sum(dim=1, keepdim=True)).sqrt()
            x = self.scale * x / rownorm_individual
        
        if self.mode == 'PN-SCS':
            rownorm_individual = (1e-6 + x.pow(2).sum(dim=1, keepdim=True)).sqrt()
            x = self.scale * x / rownorm_individual - col_mean
        
        return x
