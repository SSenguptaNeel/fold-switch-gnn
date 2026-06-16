"""
model.py
========
GCN-based binary classifier for fold-switch prediction.

Architecture
------------
Input → 3× GCNConv + BatchNorm + ReLU
      → global_mean_pool + global_max_pool  (graph-level readout)
      → 2-layer MLP head
      → sigmoid (binary output)

The edge_attr from graph_builder is not used by vanilla GCNConv.
To use edge features, swap GCNConv for GATv2Conv or NNConv (see comments).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool


class FoldSwitchGCN(nn.Module):
    """
    Parameters
    ----------
    in_channels  : node feature dimension (26)
    hidden_dim   : width of GCN layers (default 64)
    num_layers   : number of GCN message-passing layers (default 3)
    dropout      : dropout rate on MLP head (default 0.4)
    """

    def __init__(
        self,
        in_channels: int = 26,
        hidden_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.4,
    ):
        super().__init__()

        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()

        # first layer: in_channels → hidden_dim
        self.convs.append(GCNConv(in_channels, hidden_dim))
        self.bns.append(nn.BatchNorm1d(hidden_dim))

        # remaining layers: hidden_dim → hidden_dim
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        # readout: concat mean + max pooling → 2*hidden_dim
        self.head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x, edge_index, batch):
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)

        # graph-level readout
        x_mean = global_mean_pool(x, batch)   # (B, hidden_dim)
        x_max  = global_max_pool(x, batch)    # (B, hidden_dim)
        x = torch.cat([x_mean, x_max], dim=1) # (B, 2*hidden_dim)

        return self.head(x).squeeze(-1)        # (B,)  logits

    def predict_proba(self, x, edge_index, batch):
        return torch.sigmoid(self.forward(x, edge_index, batch))
