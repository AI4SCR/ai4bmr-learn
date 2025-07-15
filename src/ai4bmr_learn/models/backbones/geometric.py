import torch
import torch.nn.functional as F
from torch_geometric.nn import MeanAggregation, GCNConv


class GCN(torch.nn.Module):
    def __init__(self, num_node_features: int, embed_dim: int):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, 16)
        self.dropout = torch.nn.Dropout(0.2)
        self.conv2 = GCNConv(16, embed_dim)
        self.pool = MeanAggregation()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = self.pool(x, batch)

        return x