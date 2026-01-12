import torch.nn as nn


class MILSimple(nn.Module):

    def __init__(self, input_dim: int, pooling: str = 'mean'):
        super().__init__()

        self.input_dim = input_dim
        self.pooling = pooling
        self.output_dim = input_dim

    def forward(self, x, return_attn_and_logits: bool = False):
        assert x.dim() == 3, f'Expected input dimension 3, got {x.dim}'  # (B, N, D)
        match self.pooling:
            case 'mean':
                z =  x.mean(dim=1)
            case 'max':
                z = x.amax(dim=1)
            case _:
                raise ValueError(f'Unknown pooling method: {self.pooling}')

        if return_attn_and_logits:
            return z, None, None
        return z