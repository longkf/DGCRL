import torch
import torch_geometric
import torch.nn as nn
from torch_geometric.nn import GATConv, BatchNorm, LayerNorm, Sequential
import copy
from torch import nn
from torch.nn import Linear


class DGCRL(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, batch_norm=False, batch_norm_mm=0.99):
        super().__init__()
        self.encoder_q = Encoder(in_channels, hidden_channels, out_channels, batchnorm=batch_norm,
                                 batchnorm_mm=batch_norm_mm)
        self.mapper = Mapper(out_channels, 2 * out_channels, out_channels)
        self.encoder_k = copy.deepcopy(self.encoder_q)
        self.encoder_k.reset_parameters()
        for param in self.encoder_k.parameters():
            param.requires_grad = False

    def trainable_parameters(self):
        r"""Returns the parameters that will be updated via an optimizer."""
        return list(self.encoder_q.parameters()) + list(self.mapper.parameters())

    def update_target_network(self, mm):
        r"""Performs a momentum update of the target network's weights.

        Args:
            mm (float): Momentum used in moving average update.
        """
        assert 0.0 <= mm <= 1.0, "Momentum needs to be between 0.0 and 1.0, got %.5f" % mm
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.mul_(mm).add_(param_q.data, alpha=1. - mm)

    def forward(self, data_q, data_k):
        # forward online network
        Q = self.encoder_q(data_q)

        # prediction
        Q = self.mapper(Q)

        # forward target network
        with torch.no_grad():
            K = self.encoder_k(data_k).detach()
        return Q, K


class Mapper(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden_channels, bias=True),
            nn.PReLU(1),
            nn.Linear(hidden_channels, out_channels, bias=True)
        )
        self.reset_parameters()

    def forward(self, x):
        return self.net(x)

    def reset_parameters(self):
        # kaiming_uniform
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()


class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, batchnorm=False, batchnorm_mm=0.99):
        super().__init__()
        layers = []
        layer_sizes = [in_channels, hidden_channels, out_channels]
        for in_dim, out_dim in zip(layer_sizes[:-1], layer_sizes[1:]):
            layers.append((GATConv(in_dim, out_dim), 'x, edge_index -> x'), )
            if batchnorm:
                layers.append(BatchNorm(out_dim, momentum=batchnorm_mm))
            else:
                layers.append(LayerNorm(out_dim))
            layers.append(nn.PReLU())
        self.model = Sequential('x, edge_index', layers)

    def reset_parameters(self):
        self.model.reset_parameters()

    def forward(self, data):
        return self.model(data.x, data.edge_index)


class Decoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.lin1 = Linear(in_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, out_channels)
        self.af = nn.PReLU()

    def forward(self, Q, K, edge_label_index):
        s = Q(edge_label_index)[0]
        t = K(edge_label_index)[1]
        x = s * t
        x = self.lin1(x)
        x = self.af(x)
        x = torch.sigmoid(self.lin2(x))
        return x
