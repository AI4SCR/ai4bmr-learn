import torch.nn as nn

class Head(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.head = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        x = x.mean(dim=1)  # B, M, D
        return self.head(x)