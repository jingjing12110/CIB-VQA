# @File :bipartite_graph.py
# @Desc :https://github.com/Luoyadan/MM2020_ABG.git
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict


def cross_entropy_soft(pred):
    softmax = nn.Softmax(dim=1)
    logsoftmax = nn.LogSoftmax(dim=1)
    loss = torch.mean(torch.sum(-softmax(pred) * logsoftmax(pred), 1))
    return loss


# attentive entropy loss (source + target)
def attentive_entropy(pred, pred_domain):
    softmax = nn.Softmax(dim=1)
    logsoftmax = nn.LogSoftmax(dim=1)
    
    # attention weight
    entropy = torch.sum(-softmax(pred_domain) * logsoftmax(pred_domain), 1)
    weights = 1 + entropy
    
    # attentive entropy
    loss = torch.mean(weights * torch.sum(-softmax(pred) * logsoftmax(pred), 1))
    return loss


class BipGraphEdgeConv(nn.Module):
    def __init__(self,
                 in_features,
                 num_features,
                 ratio=(2, 1),
                 separate_dissimilarity=False,
                 dropout=0.5):
        super().__init__()
        # set size
        self.in_features = in_features
        self.num_features_list = [num_features * r for r in ratio]
        self.separate_dissimilarity = separate_dissimilarity
        self.dropout = dropout
        
        # layers
        layer_list = OrderedDict()
        for l in range(len(self.num_features_list)):
            # set layer
            layer_list['conv{}'.format(l)] = nn.Conv2d(
                in_channels=self.num_features_list[
                    l - 1] if l > 0 else self.in_features,
                out_channels=self.num_features_list[l],
                kernel_size=1,
                bias=False)
            layer_list['norm{}'.format(l)] = nn.BatchNorm2d(
                num_features=self.num_features_list[l],
            )
            layer_list['relu{}'.format(l)] = nn.LeakyReLU()
            if self.dropout > 0:
                layer_list['drop{}'.format(l)] = nn.Dropout2d(p=self.dropout)
        
        layer_list['conv_out'] = nn.Conv2d(
            in_channels=self.num_features_list[-1],
            out_channels=1,
            kernel_size=1
        )
        self.sim_network = nn.Sequential(layer_list)
    
    def forward(self, node_source, node_target):
        # [bs, n_source, 1, 768]
        x_i = node_source.unsqueeze(2)
        # [bs, 1, n_target, 768]
        x_j = node_target.unsqueeze(1)
        # source to target [bs, n_source, n_target, 768]
        x_ij = torch.abs(x_i - x_j)
        # target to source [bs, 768, n_target, n_source]
        x_ij = x_ij.permute(0, 3, 2, 1)
        
        # compute similarity/dissimilarity
        # [bs, n_target, n_source]
        sim_val = torch.sigmoid(self.sim_network(x_ij)).squeeze()
        # along source-dim -> [bs, n_source, n_target]
        source_to_target_edge = F.normalize(sim_val, p=1, dim=-1).transpose(1, 2)
        # along target-dim -> [bs, n_source, n_target]
        source_to_target_edge = F.normalize(source_to_target_edge, p=1, dim=-1)
        return source_to_target_edge


class BipGraphNodeConv(nn.Module):
    def __init__(self,
                 in_features,
                 num_features,
                 ratio=(2, 1),
                 dropout=0.0):
        super().__init__()
        # set size
        self.in_features = in_features
        self.num_features_list = [num_features * r for r in ratio]
        self.dropout = dropout
        
        # layers
        layer_list = OrderedDict()
        for l in range(len(self.num_features_list)):
            layer_list['conv{}'.format(l)] = nn.Conv2d(
                in_channels=self.num_features_list[
                    l - 1] if l > 0 else self.in_features * 2,
                out_channels=self.num_features_list[l],
                kernel_size=1,
                bias=False)
            layer_list['norm{}'.format(l)] = nn.BatchNorm2d(
                num_features=self.num_features_list[l],
            )
            layer_list['relu{}'.format(l)] = nn.LeakyReLU()
            if self.dropout > 0 and l == (len(self.num_features_list) - 1):
                layer_list['drop{}'.format(l)] = nn.Dropout(p=self.dropout)
        
        self.network = nn.Sequential(layer_list)
    
    def forward(self, node_source, node_target, source_to_target_edge):
        source_aggr_cross = torch.bmm(source_to_target_edge, node_target)
        target_aggr_cross = torch.bmm(  # [bs, 36, 768]
            source_to_target_edge.transpose(1, 2), node_source)
        
        source_node_feat = torch.cat(  # [bs, 768*2, 20]
            [node_source, source_aggr_cross], -1).transpose(1, 2)
        target_node_feat = torch.cat(  # [bs, 768*2, 36]
            [node_target, target_aggr_cross], -1).transpose(1, 2)
        
        # non-linear transform
        source_node_feat = self.network(
            source_node_feat.unsqueeze(-1)).transpose(1, 2).squeeze()
        target_node_feat = self.network(
            target_node_feat.unsqueeze(-1)).transpose(1, 2).squeeze()
        
        return source_node_feat, target_node_feat
    

class NodeUpdateNetwork(nn.Module):
    def __init__(self,
                 in_features,
                 num_features,
                 ratio=(2, 1),
                 dropout=0.0):
        super(NodeUpdateNetwork, self).__init__()
        # set size
        self.in_features = in_features
        self.num_features_list = [num_features * r for r in ratio]
        self.dropout = dropout
        
        # layers
        layer_list = OrderedDict()
        for l in range(len(self.num_features_list)):
            layer_list['conv{}'.format(l)] = nn.Conv2d(
                in_channels=self.num_features_list[
                    l - 1] if l > 0 else self.in_features * 2,
                out_channels=self.num_features_list[l],
                kernel_size=1,
                bias=False)
            layer_list['norm{}'.format(l)] = nn.BatchNorm2d(
                num_features=self.num_features_list[l],
            )
            layer_list['relu{}'.format(l)] = nn.LeakyReLU()
            if self.dropout > 0 and l == (len(self.num_features_list) - 1):
                layer_list['drop{}'.format(l)] = nn.Dropout(p=self.dropout)
        
        self.network = nn.Sequential(layer_list)
    
    def forward(self, node_source, node_target, source_to_target_edge):
        source_aggr_cross = torch.mm(source_to_target_edge, node_target)
        target_aggr_cross = torch.mm(source_to_target_edge.t(), node_source)
        # [1, 768*2, n_source]
        source_node_feat = torch.cat([node_source, source_aggr_cross],
                                     -1).unsqueeze(0).transpose(1, 2)
        # [1, 768*2, n_target]
        target_node_feat = torch.cat([node_target, target_aggr_cross],
                                     -1).unsqueeze(0).transpose(1, 2)
        
        # non-linear transform
        source_node_feat = self.network(
            source_node_feat.unsqueeze(-1)).transpose(1, 2).squeeze()
        target_node_feat = self.network(
            target_node_feat.unsqueeze(-1)).transpose(1, 2).squeeze()
        
        return source_node_feat, target_node_feat


class EdgeUpdateNetwork(nn.Module):
    def __init__(self,
                 in_features,
                 num_features,
                 ratio=(2, 1),
                 separate_dissimilarity=False,
                 dropout=0.5):
        super(EdgeUpdateNetwork, self).__init__()
        # set size
        self.in_features = in_features
        self.num_features_list = [num_features * r for r in ratio]
        self.separate_dissimilarity = separate_dissimilarity
        self.dropout = dropout

        # layers
        layer_list = OrderedDict()
        for l in range(len(self.num_features_list)):
            # set layer
            layer_list['conv{}'.format(l)] = nn.Conv2d(
                in_channels=self.num_features_list[
                    l - 1] if l > 0 else self.in_features,
                out_channels=self.num_features_list[l],
                kernel_size=1,
                bias=False)
            layer_list['norm{}'.format(l)] = nn.BatchNorm2d(
                num_features=self.num_features_list[l],
            )
            layer_list['relu{}'.format(l)] = nn.LeakyReLU()
            if self.dropout > 0:
                layer_list['drop{}'.format(l)] = nn.Dropout2d(p=self.dropout)

        layer_list['conv_out'] = nn.Conv2d(
            in_channels=self.num_features_list[-1],
            out_channels=1,
            kernel_size=1
        )
        self.sim_network = nn.Sequential(layer_list)

    def forward(self, node_source, node_target):
        x_i = node_source.unsqueeze(1)  # [n_source, 1, 768]
        x_j = torch.transpose(node_target.unsqueeze(1), 0, 1)  # [1, n_target,768]
        x_ij = torch.abs(x_i - x_j)  # source to target [n_source, n_target,768]
        # [1, 768, n_target, n_source]
        x_ij = torch.transpose(x_ij, 0, 2).unsqueeze(0)  # target to source

        # compute similarity/dissimilarity
        # [n_target, n_source]
        sim_val = torch.sigmoid(self.sim_network(x_ij)).squeeze()
        # along source-dim -> [n_source, n_target]
        source_to_target_edge = F.normalize(sim_val, p=1, dim=1).t()
        # along target-dim -> [n_source, n_target]
        source_to_target_edge = F.normalize(source_to_target_edge, p=1, dim=1)
        return source_to_target_edge


class BipGraphNetwork(nn.Module):
    def __init__(self, in_features, node_features,
                 edge_features, num_layers,
                 dropout=0.5, num_segments=0):
        super(BipGraphNetwork, self).__init__()
        # set size
        self.in_features = in_features
        self.node_features = node_features
        self.edge_features = edge_features
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_segments = num_segments
        
        # for each layer
        for l in range(self.num_layers):
            # set node to edge
            if num_segments == 0:
                node2edge_net = EdgeUpdateNetwork(
                    in_features=self.in_features if l == 0 else self.edge_features,
                    num_features=self.node_features,
                    separate_dissimilarity=False,
                    dropout=self.dropout)
                self.add_module('node2edge_net{}'.format(l), node2edge_net)
            # set edge to node
            edge2node_net = NodeUpdateNetwork(
                in_features=self.in_features if l == 0 else self.node_features,
                num_features=self.edge_features,
                dropout=self.dropout)
            
            self.add_module('edge2node_net{}'.format(l), edge2node_net)
        
        if num_segments == 0:
            print('finished constructing frame-level GNN')
        else:
            print('finished constructing video-level GNN')
    
    # forward
    def forward(self, feat_base_source, feat_base_target,
                source_target_frame_edge=None):
        
        edge_feat_list = []
        node_source_feat_list = []
        node_target_feat_list = []
        node_source = feat_base_source
        node_target = feat_base_target
        
        for l in range(self.num_layers):
            # (1) edge update
            if source_target_frame_edge is None:
                source_to_target_edge, sim_val = self._modules[
                    'node2edge_net{}'.format(l)](node_source, node_target)
            else:
                source_to_target_edge = nn.AvgPool2d(
                    kernel_size=(self.num_segments, self.num_segments))(
                    source_target_frame_edge.unsqueeze(0))
                source_to_target_edge = source_to_target_edge.squeeze()
            
            # (2) node update
            node_source, node_target = self._modules['edge2node_net{}'.format(l)](
                node_source, node_target, source_to_target_edge)
            
            # save edge feature
            edge_feat_list.append(source_to_target_edge)
            node_source_feat_list.append(node_source)
            node_target_feat_list.append(node_target)
        
        return edge_feat_list, node_source_feat_list, node_target_feat_list

