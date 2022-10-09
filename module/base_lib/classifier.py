import torch.nn as nn


class SimpleClassifier(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout):
        super(SimpleClassifier, self).__init__()
        layers = [
            nn.Linear(in_dim, hid_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hid_dim, out_dim)
        ]
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class MLP(nn.Module):
    def __init__(self, dims, n_layers, use_bn=True):
        super(MLP, self).__init__()
        self.n_layers = n_layers
        assert len(dims) == (self.n_layers + 1)
        
        self.mlp_layers = nn.ModuleList()
        for i in range(n_layers):
            if use_bn:
                self.mlp_layers.append(nn.Sequential(
                    nn.Linear(dims[i], dims[i + 1]),
                    nn.BatchNorm1d(dims[i + 1]),
                    nn.ReLU())
                )
            else:
                self.mlp_layers.append(nn.Sequential(
                    nn.Linear(dims[i], dims[i + 1]),
                    nn.ReLU())
                )
    
    def forward(self, x):
        """
        :param x: [bs, *, dim]
        :return:
        """
        for layer in range(self.n_layers):
            x = self.mlp_layers[layer](x)
        return x


class MultiConv1x1(nn.Module):
    def __init__(self, channels, n_layers, use_bn=True):
        super(MultiConv1x1, self).__init__()
        self.n_layers = n_layers
        assert len(channels) == (self.n_layers + 1)
        
        self.conv_layers = nn.ModuleList()
        for i in range(n_layers):
            if use_bn:
                self.conv_layers.append(nn.Sequential(
                    nn.Conv2d(channels[i], channels[i + 1], kernel_size=1),
                    nn.BatchNorm2d(channels[i + 1]),
                    nn.ReLU(inplace=True))
                )
            else:
                self.conv_layers.append(nn.Sequential(
                    nn.Conv2d(channels[i], channels[i + 1], kernel_size=1),
                    nn.ReLU(inplace=True))
                )
    
    def forward(self, x):
        """
        :param x: [bs, C, W, H]
        :return:
        """
        for layer in range(self.n_layers):
            x = self.conv_layers[layer](x)
        return x
