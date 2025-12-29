import lightning.pytorch as L

import torch
import torch.nn as nn

from ai4bmr_learn.supervised.abmil import MIL


class MuMoMIL(L.LightningModule):
    def __init__(self,
                 hne_dim: int, xe_dim: int,
                 mode: str = 'concat',  # 'add'
                 layer_norm: bool = False,
                 **mil_kwargs,
                 ):
        super().__init__()

        self.mode = mode

        self.hne_mil = MIL(input_dim=hne_dim, **mil_kwargs)
        self.xe_mil = MIL(input_dim=xe_dim, **mil_kwargs)

        self.hne_layer_norm = nn.LayerNorm(hne_dim) if layer_norm else nn.Identity()
        self.xe_layer_norm = nn.LayerNorm(xe_dim) if layer_norm else nn.Identity()

        match self.mode:
            case 'concat':
                self.output_dim = self.hne_mil.output_dim + self.xe_mil.output_dim
            case 'add':
                assert self.hne_mil.output_dim == self.xe_mil.output_dim, f'For add mode, hne output dim {self.hne_mil.output_dim} must equal xe output dim {self.xe_mil.output_dim}'
                self.output_dim = self.hne_mil.output_dim
            case _:
                raise ValueError(f'Unknown mode {self.mode}')

    def forward(self, item, return_attn_and_logits: bool = False):
        hne = item['modalities']['hne']['x']
        xe = item['modalities']['xe']['x']

        z1, attn1, logits1 = self.hne_mil(hne, return_attn_and_logits=True)
        z2, attn2, logits2 = self.xe_mil(xe, return_attn_and_logits=True)

        z1 = self.hne_layer_norm(z1)
        z2 = self.xe_layer_norm(z2)

        match self.mode:
            case 'concat':
                z = torch.cat([z1, z2], dim=-1)
            case 'add':
                z = z1 + z2
            case _:
                raise ValueError(f'Unknown mode {self.mode}')

        if return_attn_and_logits:
            attn = {'hne': attn1.detach().cpu(), 'xe': attn2.detach().cpu()}
            logits = {'hne': logits1.detach().cpu(), 'xe': logits2.detach().cpu()}
            return z, attn, logits
        else:
            return z
