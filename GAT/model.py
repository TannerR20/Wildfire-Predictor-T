"""Two spatial graph-attention layers and a per-node binary logit."""
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv


class WildfireGAT(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, heads=4, dropout=0.2):
        super().__init__()
        self.dropout = dropout
        self.gat1 = GATConv(input_dim, hidden_dim, heads=heads, dropout=dropout)
        self.gat2 = GATConv(hidden_dim * heads, hidden_dim, heads=1,
                            concat=False, dropout=dropout)
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.gat1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.gat2(x, edge_index))
        # BCEWithLogitsLoss applies sigmoid internally during training.
        return self.output(x).squeeze(-1)
