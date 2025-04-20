import torch
import torch.nn as nn
from pydantic import BaseModel
from sklearn.neighbors import radius_neighbors_graph
from timm.layers import trunc_normal_
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv

from ai4bmr_learn.models.encoder.base import BaseEncoder


class GraphEncoderConfig(BaseModel):
    dim: int = 192
    num_layers: int = 8
    num_heads: int = 3
    mlp_dim: int = 768
    dropout: float = 0.1  # applied on a levels (attn, mlp, etc.)
    num_prefix_tokens: int = 1

class GraphEncoder(BaseEncoder):

    def __init__(self, radius: int = 36, in_channels: int = 3, dim: int = 192, num_layers=8, num_heads=6, mlp_dim: int = 768, dropout: float = 0.1):
        super().__init__()

        self.num_prefix_tokens = 1

        self.radius = radius
        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, dim))

        self.conv1 = GCNConv(in_channels, dim // 2)
        self.conv2 = GCNConv(dim // 2, dim)
        self.layer_norm = nn.LayerNorm(dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)


    def forward(self, batch):
        data, coords, num_samples = batch['data'], batch['coords'], batch['num_samples']
        data, coords = torch.split(data, num_samples, dim=0), torch.split(coords, num_samples, dim=0)
        x = self.build_graph(data, coords)
        x, edge_index = x.x, x.edge_index

        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()

        # NOTE: we have the problem that we need to be able to handle different sequence lengths
        #   we could achieve this by using
        #   from xformers.ops import fmha;
        #   fmha.BlockDiagonalMask.from_tensor_list(list_x)

        x = self.layer_norm(x)
        xs = x.split(num_samples)
        x = self.pos_embed(xs[0])
        x = self.transformer(x)
        return x


    def build_graph(self, data, coords) -> Batch:
        list_of_graphs = []
        for data_, coords_ in zip(data, coords):
            adj = radius_neighbors_graph(coords_, radius=self.radius, mode='connectivity', include_self=False)
            adj = torch.tensor(adj.toarray())
            edge_index = adj.nonzero(as_tuple=False).t().contiguous().long()

            graph = Data(x=data_.float(), edge_index=edge_index)
            graph.validate(raise_on_error=True)
            list_of_graphs.append(graph)

        batch = Batch.from_data_list(list_of_graphs)
        return batch

    def pos_embed(self, x):
        # TODO:
        #   - absolute positional embedding
        #   - learnable positional embedding
        #   - relative positional embedding
        batch_size = x.shape[0]

        # x = x + self.pos_embedding
        x = torch.cat([self.cls_token.expand(1, batch_size, -1), x], dim=1)

        return x

    def init_weight(self):
        trunc_normal_(self.cls_token, std=0.02)
        # trunc_normal_(self.pos_embedding, std=0.02)

coords = torch.cat([torch.randn(5, 2), torch.randn(4, 2)]) * 18
data = torch.cat([torch.randn(5, 3), torch.randn(4, 3)])
num_samples = [5,4]
batch = {'data': data, 'coords': coords, 'num_samples': num_samples}

graph = GraphEncoder()
out = graph(batch)
