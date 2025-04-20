import torch
from einops import rearrange
from pydantic import BaseModel, ConfigDict, Field
from timm.layers import trunc_normal_
from timm.models.vision_transformer import Block

from ai4bmr_learn.models.utils import get_at_index


# class EncoderConfig(BaseModel):
#     # NOTE: we do not consider image_size, patch_size, num_channels, num_classes as part of the pure encoder
#     #   however, we include them here for end-to-end model construction
#     image_size: int = 224
#     patch_size: int = 16
#     num_channels: int = 3
#     num_classes: int = 1000
#     token_dim: int = 1024  # timm: 768
#     depth: int = 6  # timm: 12
#     num_head: int = 12  # lucidrain: 16
#     mlp_dim: int = 2048  # timm: 3072, i.e. mlp_ratio=4
#     embed_dropout: float = 0.1  # timm: 0.0
#     attn_dropout: float = 0.1  # timm: 0.0


class EncoderDefault(BaseModel):
    num_patches: int = None
    dim: int = 192
    num_layers: int = 12
    num_heads: int = 3
    num_prefix_tokens: int = 1


class EncoderConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = "default"
    init_kwargs: EncoderDefault = Field(default_factory=EncoderDefault)


from ai4bmr_learn.models.encoder.base import BaseMaskedEncoder


class MaskedEncoder(BaseMaskedEncoder):
    def __init__(
            self,
            num_patches: int,
            dim: int,
            num_layers: int,
            num_heads: int,
            num_prefix_tokens: int = 1,
            pos_embed_prefix_tokens: bool = False,
    ) -> None:

        self.dim = dim
        self.num_patches = num_patches
        self.num_prefix_tokens = num_prefix_tokens

        super().__init__(num_tokens=num_patches + num_prefix_tokens)

        self.pos_embed_prefix_tokens = pos_embed_prefix_tokens

        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, dim))

        if pos_embed_prefix_tokens:
            self.pos_embedding = torch.nn.Parameter(
                torch.zeros(num_patches + num_prefix_tokens, 1, dim)
            )
        else:
            self.pos_embedding = torch.nn.Parameter(torch.zeros(num_patches, 1, dim))

        self.transformer = torch.nn.Sequential(
            *[Block(dim, num_heads) for _ in range(num_layers)]
        )

        self.layer_norm = torch.nn.LayerNorm(dim)

        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.cls_token, std=0.02)
        trunc_normal_(self.pos_embedding, std=0.02)

    def pos_embed(self, x):
        batch_size = x.shape[0]

        x = rearrange(x, "b n d -> n b d")

        if self.pos_embed_prefix_tokens:
            x = torch.cat([self.cls_token.expand(1, batch_size, -1), x], dim=0)
            x = x + self.pos_embedding
        else:
            x = x + self.pos_embedding
            x = torch.cat([self.cls_token.expand(1, batch_size, -1), x], dim=0)

        x = rearrange(x, "n b d -> b n d")
        return x

    def forward(self, x):
        x = self.pos_embed(x)
        x = self.layer_norm(self.transformer(x))
        return x

    def forward_masked(self, x, idx_keep: torch.Tensor):
        x = self.pos_embed(x)
        x = get_at_index(x, index=idx_keep.to(x.device))
        x = self.layer_norm(self.transformer(x))
        return x
