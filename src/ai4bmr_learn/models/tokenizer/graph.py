import torch
import torch.nn as nn
from ai4bmr_learn.models.tokenizer.base import BaseTokenizer
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

class TokenizerGraph(BaseTokenizer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.conv1 = GCNConv(self.in_channels, self.dim // 2)
        self.conv2 = GCNConv(self.dim // 2, self.dim)
        self.norm = nn.LayerNorm(self.dim)

    def forward(self, graph: Data) -> torch.Tensor:
        x, edge_index = graph.x, graph.edge_index

        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        x = self.norm(x)

        return x

