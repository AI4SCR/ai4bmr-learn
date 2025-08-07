import torch.nn as nn
import torch

class Linear(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, aggregation: str = 'mean'):
        super().__init__()
        self.head = nn.Linear(input_dim, num_classes)

        match aggregation:
            case 'mean':
                self.agg = torch.mean
            case 'max':
                self.agg = torch.max
            case 'min':
                self.agg = torch.min

    def forward(self, x):
        x = self.agg(x, dim=1)  # B, M, D
        return self.head(x)