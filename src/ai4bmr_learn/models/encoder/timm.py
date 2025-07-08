# check: resample_abs_pos_embed in timm
import torch
from ai4bmr_learn.models.tokenizer.base import BaseTokenizer
from ai4bmr_learn.models.encoder.masked_encoder import BaseMaskedEncoder


from ai4bmr_learn.models.utils import get_at_index


class Tokenizer(BaseTokenizer):
    def __init__(self, model, image_size: int):

        kernel_size = model.patch_embed.proj.kernel_size
        dim = model.patch_embed.proj.out_channels
        num_channels = model.patch_embed.proj.in_channels

        super().__init__(
            image_size=image_size,
            kernel_size=kernel_size,
            dim=dim,
            num_channels=num_channels,
        )

        self.model = model.patch_embed

        assert self.model.proj.kernel_size == self.kernel_size
        assert self.model.proj.stride == self.stride

    def forward(self, x):
        return self.model(x)

    def tokens2img(self, x: torch.Tensor) -> torch.Tensor:
        from einops import rearrange

        kh, kw = self.kernel_size
        h, w = self.grid_size
        img = rearrange(x, "b (h w) (c p1 p2) -> b c (h p1) (w p2)", p1=kh, p2=kw, h=h, w=w)
        return img

class MaskedEncoder(BaseMaskedEncoder):
    def __init__(self, model, num_patches: int):
        self.num_patches = num_patches
        self.num_prefix_tokens = model.num_prefix_tokens
        self.dim = model.embed_dim

        super().__init__(num_tokens=self.num_patches + self.num_prefix_tokens)

        self.model = model

        # remove unused layers
        # del self.model.patch_embed  # needed for dynamic_image_size = True
        del self.model.fc_norm
        del self.model.head_drop
        del self.model.head

    def forward_encoder(self, x):
        x = self.model.patch_drop(x)
        x = self.model.norm_pre(x)
        x = self.model.blocks(x)
        x = self.model.norm(x)
        return x

    def forward(self, x):
        x = self.model._pos_embed(x)
        x = self.forward_encoder(x)
        return x

    def forward_masked(self, x, idx_keep: torch.Tensor):
        x = self.model._pos_embed(x)
        x = get_at_index(x, index=idx_keep.to(x.device))
        x = self.forward_encoder(x)
        return x

import torch.nn as nn
import timm
class Backbone(nn.Module):

    def __init__(self,
                 model_name: str = 'vit_small_patch16_224',
                 num_classes: int = 0,
                 global_pool: str = "token",
                 image_size: int = 224,
                 dynamic_img_size: bool = True,
                 num_channels: int = 3,
                 pretrained: bool = False
                 ):
        super().__init__()

        model = timm.create_model(model_name=model_name,
                                     num_classes=num_classes,
                                     global_pool=global_pool,
                                     img_size=image_size,
                                     dynamic_img_size=dynamic_img_size,
                                     in_chans=num_channels,
                                     pretrained=pretrained)

        tokenizer = Tokenizer(model, image_size=image_size)
        encoder = MaskedEncoder(model, num_patches=tokenizer.num_tokens)