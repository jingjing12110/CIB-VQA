# @File :gcn.py 
# @Time :2020/8/17 
# @Email :jingjingjiang2017@gmail.com
import math
import torch
import torch.nn as nn
import torch.nn.functional as F



class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
    

class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, act, bias=True):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU() if act == 'prelu' else act
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)
        
        for m in self.modules():
            self.weights_init(m)
    
    @staticmethod
    def weights_init(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)
    
    def forward(self, seq, adj, sparse=False):
        """
        @param seq: [bs, n_node, n_feature]
        @param adj: [bs, n_node, n_node]
        @param sparse: bool
        @return:
        """
        seq_fts = self.fc(seq)
        if sparse:
            out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq_fts, 0)), 0)
        else:
            out = torch.bmm(adj, seq_fts)
        if self.bias is not None:
            out += self.bias
        return self.act(out)


class GCNConv(nn.Module):
    def __init__(self, dim_hidden, dropout=0.0):
        super(GCNConv, self).__init__()
        self.ctx_layer = nn.Linear(dim_hidden, dim_hidden, bias=False)
        self.layer_norm = nn.LayerNorm(dim_hidden)
        self.dropout = nn.Dropout(p=dropout)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.ctx_layer.weight)

    def forward(self, x, adj):
        """
        @param x: (bs, num_nodes, embed_size)
        @param adj: (bs, num_nodes, num_nodes)
        @return:
        """
        node_embeds = x + self.dropout(torch.bmm(adj, self.ctx_layer(x)))
        return self.layer_norm(node_embeds)


class EnsembleGCN(nn.Module):
    def __init__(self, input_dim, hidden_dims, n_layers, dropout=0.5):
        super(EnsembleGCN, self).__init__()
        self.dropout_p = dropout

        self.gnn_layers = nn.ModuleList()
        self.linear_prediction = nn.ModuleList()
        for i in range(n_layers):
            if i == 0:
                self.gnn_layers.append(
                    GCNConv(input_dim)
                )
                self.linear_prediction.append(
                    nn.Sequential(nn.Linear(input_dim, hidden_dims[i]),
                                  nn.LayerNorm(hidden_dims[i]),
                                  nn.ReLU(inplace=True)))
            else:
                self.gnn_layers.append(
                    GCNConv(hidden_dims[i - 1])
                )
                self.linear_prediction.append(
                    nn.Sequential(nn.Linear(hidden_dims[i - 1], hidden_dims[i]),
                                  nn.LayerNorm(hidden_dims[i]),
                                  nn.ReLU(inplace=True)))
        self.linear_prediction.append(
            nn.Sequential(nn.Linear(hidden_dims[-2], hidden_dims[-1]),
                          nn.LayerNorm(hidden_dims[-1]),
                          nn.ReLU(inplace=True)))

    def forward(self, x, adj):
        hidden_states = [x]
        for layer in self.gnn_layers:
            x = layer(x, adj)
            hidden_states.append(x)

        ret = 0.
        for layer, h in enumerate(hidden_states):
            ret = ret + F.dropout(
                self.linear_prediction[layer](h),
                self.dropout_p,
                training=self.training
            )
        return ret

