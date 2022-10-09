# @File :dgi.py
# @Github :https://github.com/PetarV-/DGI
import torch
import torch.nn as nn

from module.graph_lib.gcn import GCN


class AvgReadout(nn.Module):
    def __init__(self):
        """Applies an average on seq, of shape (batch, nodes, features)
        while taking into account the masking of msk
        """
        super(AvgReadout, self).__init__()
    
    def forward(self, seq, msk):
        if msk is None:
            return torch.mean(seq, 1)
        else:
            msk = torch.unsqueeze(msk, -1)
            return torch.sum(seq * msk, 1) / torch.sum(msk)


class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)
        
        for m in self.modules():
            self.weights_init(m)
    
    @staticmethod
    def weights_init(m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)
    
    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        c_x = torch.unsqueeze(c, 1)
        c_x = c_x.expand_as(h_pl)
        
        sc_1 = torch.squeeze(self.f_k(h_pl, c_x), 2)
        sc_2 = torch.squeeze(self.f_k(h_mi, c_x), 2)
        
        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2
        
        logits = torch.cat((sc_1, sc_2), 1)
        
        return logits


class DGI(nn.Module):
    def __init__(self, n_in, n_h, activation):
        """Deep Graph Infomax (Veličković et al., ICLR 2019)
        """
        super(DGI, self).__init__()
        self.gcn = GCN(n_in, n_h, activation)
        self.read = AvgReadout()
        self.sigm = nn.Sigmoid()
        self.disc = Discriminator(n_h)
    
    def forward(self, seq1, seq2, adj, sparse, msk, samp_bias1, samp_bias2):
        """
        @param seq1:
        @param seq2:
        @param adj:
        @param sparse:
        @param msk:
        @param samp_bias1:
        @param samp_bias2:
        @return:
        """
        h_1 = self.gcn(seq1, adj, sparse)
        
        c = self.read(h_1, msk)
        c = self.sigm(c)
        
        h_2 = self.gcn(seq2, adj, sparse)
        ret = self.disc(c, h_1, h_2, samp_bias1, samp_bias2)
        return ret
    
    # Detach the return variables
    def embed(self, seq, adj, sparse, msk):
        h_1 = self.gcn(seq, adj, sparse)
        c = self.read(h_1, msk)
        return h_1.detach(), c.detach()
