import torch
from einops import rearrange, repeat
from pydantic import BaseModel
from timm.layers import trunc_normal_
from timm.models.vision_transformer import Block

from ai4bmr_learn.models.decoder.base import BaseMaskedDecoder
from ai4bmr_learn.models.utils import set_at_index


class MaskedDecoderDefault(BaseModel):
    num_tokens: int
    dim: int = 192
    num_layers: int = 12
    num_heads: int = 3


class MaskedDecoder(BaseMaskedDecoder):
    def __init__(
        self,
        num_tokens: int,
        dim: int,
        num_layers: int,
        num_heads: int,
    ) -> None:

        super().__init__(dim=dim)

        self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embedding = torch.nn.Parameter(torch.zeros(num_tokens, 1, dim))
        self.transformer = torch.nn.Sequential(
            *[Block(dim, num_heads) for _ in range(num_layers)]
        )

        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.mask_token, std=0.02)
        trunc_normal_(self.pos_embedding, std=0.02)

    def pos_embed(self, x):
        x = rearrange(x, "b n d -> n b d")
        x = x + self.pos_embedding
        x = rearrange(x, "n b d -> b n d")
        return x

    def forward(self, x):
        x = self.pos_embed(x)
        x = self.transformer(x)
        return x

    def forward_masked(self, x, idx_keep: torch.Tensor):
        batch_size = x.shape[0]
        num_tokens = self.pos_embedding.shape[0]

        features = repeat(self.mask_token, "1 1 d -> b n d", b=batch_size, n=num_tokens)
        features = features.to(x.device)
        features = set_at_index(features, index=idx_keep, value=x)

        features = self.pos_embed(features)
        features = self.transformer(features)

        return features
