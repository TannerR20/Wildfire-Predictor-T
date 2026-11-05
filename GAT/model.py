
from torch import Tensor, nn
from torch.nn import functional as F
from torch_geometric.nn import GATConv


class GAT(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        hidden_channels: int = 8,
        heads: int = 8,
        dropout: float = 0.6,
    ) -> None:
        super().__init__()
        self.dropout = dropout

        # Each head learns how to weight incoming neighbors. The layer
        # normalizes scores over a node's incoming edges, including its self-loop.
        self.attention = GATConv(
            in_channels,
            hidden_channels,
            heads=heads,
            concat=True,
            dropout=dropout,
            add_self_loops=True,
        )
        # Concatenation above produces heads * hidden_channels per node.
        self.classifier = GATConv(
            hidden_channels * heads,
            num_classes,
            heads=1,
            concat=False,
            dropout=dropout,
            add_self_loops=True,
        )

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        # Feature dropout here is separate from attention dropout in GATConv.
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.attention(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        # Raw logits: [num_nodes, num_classes]. Cross-entropy applies log-softmax.
        return self.classifier(x, edge_index)

